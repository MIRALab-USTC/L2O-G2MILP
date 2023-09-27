from typing import Tuple, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.nn import (GraphNorm, JumpingKnowledge, MessagePassing,
                                aggr)
from torch_geometric.utils import scatter

from src.data import BipartiteGraph


class BipartiteGraphEmbedding(nn.Module):
    def __init__(self, config: DictConfig) -> None:
        """
        Embed the bipartite graph into the embedding space.
        """
        super().__init__()
        self.emb_size = config.embd_size
        self.hidden_size = config.hidden_size
        self.cons_nfeats = 1
        self.edge_nfeats = 1
        self.var_nfeats = 9

        self.cons_embedding = nn.Sequential(
            nn.Linear(self.cons_nfeats, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.emb_size)
        )

        self.edge_embedding = nn.Sequential(
            nn.Linear(self.edge_nfeats, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.emb_size)
        )

        self.var_embedding = nn.Sequential(
            nn.Linear(self.var_nfeats, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.emb_size)
        )

        self.mask_embedding = nn.Parameter(
            torch.randn(self.emb_size), requires_grad=True)

    def embed_graph(self, graph: Union[Batch, BipartiteGraph]) -> Union[Batch, BipartiteGraph]:
        """
        Embed the bipartite graphs into the embedding space.

        Input:
            graph: a batch of bipartite graphs

        Output:
            graph: a batch of bipartite graphs with embedded features
        """
        graph = graph.clone()
        graph.x_constraints = self.cons_embedding.forward(graph.x_constraints)
        graph.edge_attr = self.edge_embedding.forward(graph.edge_attr)
        graph.x_variables = self.var_embedding.forward(graph.x_variables)
        return graph

    def mask_and_embed_graph(self, graph: Union[Batch, BipartiteGraph]) -> Union[Batch, BipartiteGraph]:
        """
        Randomly mask a constraint node of each input graph. Then embed the masked bipartite graphs into the embedding space.

        Input:
            graph: a batch of bipartite graphs

        Output:
            graph: a batch of masked bipartite graphs with embedded features
        """
        if isinstance(graph, list):
            graph_list = graph.copy()
        else:
            graph_list = graph.to_data_list()

        for bid, graph in enumerate(graph_list):
            # get the masked constraint idx
            masked_cons_idx = torch.randint(
                0, graph.num_constraints, (1,)).item()

            # get the labels
            masked_edges_idx = torch.where(
                graph.edge_index[0] == masked_cons_idx)[0]
            connected_vars_idx = graph.edge_index[1][masked_edges_idx]
            logits_label = torch.zeros(graph.num_variables, dtype=int)
            logits_label[connected_vars_idx] = 1
            degree_label = torch.sum(logits_label)

            device = graph.x_constraints.device
            graph.masked_cons_idx = masked_cons_idx
            graph.bias_label = graph.x_constraints[masked_cons_idx].to(device)
            graph.degree_label = degree_label.view(1).to(device)
            graph.logits_label = logits_label.to(device)
            graph.connected_vars_idx = connected_vars_idx.to(device)
            graph.weights_label = graph.edge_attr[masked_edges_idx].to(device)

            # update the graph
            graph = self.embed_graph(graph)
            seen_edges_idx = torch.where(
                graph.edge_index[0] != masked_cons_idx)[0]
            graph.edge_index = graph.edge_index[:, seen_edges_idx]
            graph.edge_attr = graph.edge_attr[seen_edges_idx]
            graph.x_constraints[masked_cons_idx] = self.mask_embedding

            graph_list[bid] = graph

        return Batch.from_data_list(graph_list, follow_batch=["x_constraints", "x_variables"])


class BipartiteGraphGNN(nn.Module):
    def __init__(self,
                 config: DictConfig,
                 is_masked: bool = False
                 ):
        """
        The GNN model for bipartite graphs.

        Args:
            config: the configuration of the GNN model
            is_masked: whether the input graph is masked
        """
        super().__init__()
        self.emb_size = config.embd_size
        self.hidden_size = config.hidden_size
        self.depth = config.depth
        self.cons_nfeats = 1
        self.edge_nfeats = 1
        self.var_nfeats = 9
        self.jk = config.jk
        self.is_masked = is_masked

        self.conv_v_to_c_layers = nn.Sequential()
        self.conv_c_to_v_layers = nn.Sequential()
        self.graph_norm_v_to_c_layers = nn.Sequential()
        self.graph_norm_c_to_v_layers = nn.Sequential()

        for _ in range(self.depth):
            self.conv_v_to_c_layers.append(BipartiteGraphConvolution(
                config, aggr_coef=config.aggr_coef.v_to_c))
            self.graph_norm_v_to_c_layers.append(GraphNorm(self.emb_size))
            self.conv_c_to_v_layers.append(BipartiteGraphConvolution(
                config, aggr_coef=config.aggr_coef.c_to_v))
            self.graph_norm_c_to_v_layers.append(GraphNorm(self.emb_size))

        if self.is_masked:
            self.virtual_c_to_v_layers = nn.Sequential()
            self.virtual_v_to_c_layers = nn.Sequential()
            for _ in range(self.depth):
                self.virtual_v_to_c_layers.append(VirtualAggr(config))
                self.virtual_c_to_v_layers.append(nn.Sequential(
                    nn.Linear(2 * self.emb_size, self.hidden_size),
                    nn.ReLU(),
                    nn.Linear(self.hidden_size, self.emb_size)
                ))

        if self.jk == 'cat':
            self.var_jk_layer = nn.Sequential(
                JumpingKnowledge(
                    mode=self.jk, channels=self.emb_size, num_layers=self.depth + 1),
                nn.Linear(self.emb_size * (self.depth + 1), self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.emb_size)
            )
            self.cons_jk_layer = nn.Sequential(
                JumpingKnowledge(
                    mode=self.jk, channels=self.emb_size, num_layers=self.depth + 1),
                nn.Linear(self.emb_size * (self.depth + 1), self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.emb_size)
            )
        elif self.jk == 'lstm':
            self.var_jk_layer = JumpingKnowledge(
                mode=self.jk, channels=self.emb_size, num_layers=self.depth + 1)
            self.cons_jk_layer = JumpingKnowledge(
                mode=self.jk, channels=self.emb_size, num_layers=self.depth + 1)

    def forward(self, graph: BipartiteGraph) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the GNN model.

        Input:
            graph: a batch of bipartite graphs

        Output:
            h_conss: the final embedding of constraint nodes
            h_vars: the final embedding of variable nodes
        """
        constraint_features = graph.x_constraints
        edge_indices = graph.edge_index
        edge_features = graph.edge_attr
        variable_features = graph.x_variables

        constraint_features_batch = graph.x_constraints_batch
        variable_features_batch = graph.x_variables_batch

        reversed_edge_indices = torch.stack(
            [edge_indices[1], edge_indices[0]], dim=0)

        if self.jk:
            constraint_features_list, variable_features_list = [], []
            constraint_features_list.append(constraint_features)
            variable_features_list.append(variable_features)

        for i in range(self.depth):
            constraint_features = self.conv_v_to_c_layers[i](
                variable_features, reversed_edge_indices, edge_features, constraint_features
            )
            variable_features = self.conv_c_to_v_layers[i](
                constraint_features, edge_indices, edge_features, variable_features
            )
            if self.is_masked and hasattr(graph, "masked_cons_idx"):
                virtual_aggr_info = self.virtual_v_to_c_layers[i](
                    variable_features, constraint_features[graph.masked_cons_idx], graph.x_variables_batch
                )
                constraint_features[graph.masked_cons_idx] = virtual_aggr_info
                variable_features = self.virtual_c_to_v_layers[i](
                    torch.cat(
                        (variable_features, virtual_aggr_info[graph.x_variables_batch]), dim=-1)
                )

            constraint_features = self.graph_norm_v_to_c_layers[i](
                constraint_features, constraint_features_batch)
            variable_features = self.graph_norm_c_to_v_layers[i](
                variable_features, variable_features_batch)

            if self.jk:
                constraint_features_list.append(constraint_features)
                variable_features_list.append(variable_features)

        if self.jk:
            h_conss = self.cons_jk_layer(constraint_features_list)
            h_vars = self.var_jk_layer(variable_features_list)
        else:
            h_conss = constraint_features
            h_vars = variable_features

        return h_conss, h_vars


class BipartiteGraphConvolution(MessagePassing):
    def __init__(self, config, aggr_coef=None):
        if config.aggr in ["add", "mean", "max", "min"]:
            super().__init__(aggr=config.aggr)
        elif config.aggr == "softmax":
            super().__init__(aggr=aggr.SoftmaxAggregation(learn=True))
        else:
            raise NotImplementedError

        self.emb_size = config.embd_size
        self.hidden_size = config.hidden_size
        self.aggr_coef = aggr_coef

        self.feature_module_left = nn.Sequential(
            nn.Linear(self.emb_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.emb_size)
        )

        self.feature_module_edge = nn.Sequential(
            nn.Linear(self.emb_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.emb_size)
        )

        self.feature_module_right = nn.Sequential(
            nn.Linear(self.emb_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.emb_size)
        )

        self.feature_module_final = nn.Sequential(
            nn.Linear(self.emb_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.emb_size)
        )

        self.output_module = nn.Sequential(
            nn.Linear(2 * self.emb_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.emb_size)
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        output = self.propagate(
            edge_indices,
            size=(left_features.shape[0], right_features.shape[0]),
            node_features=(left_features, right_features),
            edge_features=edge_features,
        )
        return self.output_module(
            torch.cat([output, right_features], dim=-1)
        )

    def message(self, node_features_i, node_features_j, edge_features):
        output = self.feature_module_final(
            self.feature_module_left(node_features_i)
            + self.feature_module_edge(edge_features)
            + self.feature_module_right(node_features_j)
        )
        return output


class VirtualAggr(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.output_module = nn.Sequential(
            nn.Linear(2 * config.embd_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, config.embd_size)
        )

    def forward(
        self,
        x_variables: Tensor,
        x_virtual_constraints: Tensor,
        x_variables_batch: Tensor,
    ):
        x = scatter(x_variables, x_variables_batch, reduce='mean', dim=0)
        return self.output_module(
            torch.cat([x, x_virtual_constraints], dim=-1)
        )
