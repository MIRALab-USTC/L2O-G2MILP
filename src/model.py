import json
from typing import List, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.utils import sort_edge_index

import src.tb_writter as tb_writter
from src.data import BipartiteGraph
from src.modules import (Bias_Predictor, Degree_Predictor, Logits_Predictor,
                         ReSample, Weights_Predictor)
from src.nn import BipartiteGraphEmbedding, BipartiteGraphGNN


class G2MILP(nn.Module):

    def __init__(self,
                 config: DictConfig,
                 data_stats: dict
                 ):
        """
        G2MILP model, the main model.

        Args:
            config: model config
            data_stats: dataset statistics, used to configure the model
        """
        super().__init__()

        self.config = config

        self.embedding_layer = BipartiteGraphEmbedding(config.graph_embedding)
        self.encoder_layer = BipartiteGraphGNN(config.gnn)
        self.decoder_layer = BipartiteGraphGNN(config.gnn, is_masked=True)
        self.resample_layer = ReSample(config.resample)

        self.bias_predictor = Bias_Predictor(
            config.bias_predictor,
            data_stats,
        )
        self.degree_predictor = Degree_Predictor(
            config.degree_predictor,
            data_stats,
        )
        self.logits_predictor = Logits_Predictor(
            config.logits_predictor,
            data_stats,
        )
        self.weights_predictor = Weights_Predictor(
            config.weights_predictor,
            data_stats,
        )

    def forward(self,
                data: Union[Batch, BipartiteGraph],
                beta_cons: float,
                beta_var: float
                ) -> Tensor:
        """
        Forward pass of the model (for training).

        Inputs:
            data: batch of bipartite graphs
            beta_cons: coefficient of the KL loss for constraints
            beta_var: coefficient of the KL loss for variables

        Outputs:
            loss: loss of the model
        """
        orig_graph = self.embedding_layer.embed_graph(data)
        masked_graph = self.embedding_layer.mask_and_embed_graph(data)

        z_conss, z_vars = self.encoder_layer.forward(orig_graph)
        z_conss, z_vars, cons_kl_loss, var_kl_loss = self.resample_layer.forward(
            z_conss, z_vars)

        h_conss, h_vars = self.decoder_layer.forward(masked_graph)

        cons_loss, cons_pred = self.bias_predictor.forward(
            z_conss, h_conss, masked_graph.masked_cons_idx, masked_graph.bias_label)

        degree_loss, degree_pred = self.degree_predictor.forward(
            z_conss, h_conss, masked_graph.masked_cons_idx, masked_graph.degree_label)

        logits_loss, logits_pred = self.logits_predictor.forward(
            z_vars, h_vars, masked_graph.logits_label)

        weights_loss, weights_pred = self.weights_predictor.forward(
            z_vars, h_vars, masked_graph.connected_vars_idx, masked_graph.weights_label)

        cons_loss = cons_loss * self.config.loss_weights.cons_loss
        degree_loss = degree_loss * self.config.loss_weights.degree_loss
        logits_loss = logits_loss * self.config.loss_weights.logits_loss
        weights_loss = weights_loss * self.config.loss_weights.weights_loss
        loss = (beta_cons * cons_kl_loss + beta_var * var_kl_loss + cons_loss +
                degree_loss + logits_loss + weights_loss) / data.num_graphs

        tb = tb_writter.tb_writter
        if self.training:
            tb.add_histogram("Embeddings/h_vars", h_vars, tb_writter.step)
            tb.add_histogram("Embeddings/h_conss", h_conss, tb_writter.step)
            tb.add_histogram("Embeddings/z_vars", z_vars, tb_writter.step)
            tb.add_histogram("Embeddings/z_conss", z_conss, tb_writter.step)
        else:
            tb.add_histogram("Embeddings/h_vars_val", h_vars, tb_writter.step)
            tb.add_histogram("Embeddings/h_conss_val",
                             h_conss, tb_writter.step)
            tb.add_histogram("Embeddings/z_vars_val", z_vars, tb_writter.step)
            tb.add_histogram("Embeddings/z_conss_val",
                             z_conss, tb_writter.step)

        return loss

    def decode(self, graphs: List[BipartiteGraph], config: dict):

        avg_num_constraints = sum([len(graph.x_constraints)
                                  for graph in graphs]) / len(graphs)
        num_iters = round(avg_num_constraints * config.mask_ratio)
        for _ in range(num_iters):

            masked_graphs = self.embedding_layer.mask_and_embed_graph(graphs)

            z_conss = torch.randn(
                (len(masked_graphs.x_constraints_batch), self.config.common.embd_size)).cuda()
            z_vars = torch.randn(
                (len(masked_graphs.x_variables_batch), self.config.common.embd_size)).cuda()

            h_conss, h_vars = self.decoder_layer(masked_graphs)

            conss = self.bias_predictor.decode(
                z_conss, h_conss, masked_graphs.masked_cons_idx)

            degree = self.degree_predictor.decode(
                z_conss, h_conss, masked_graphs.masked_cons_idx)

            weights_idx = self.logits_predictor.decode(
                z_vars, h_vars, masked_graphs.x_variables_batch, degree)

            weights = self.weights_predictor.decode(
                z_vars, h_vars, masked_graphs.x_variables_batch, weights_idx)

            masked_graphs = masked_graphs.to_data_list()
            weights_ptr = 0
            for bid, (graph, masked_graph) in enumerate(zip(graphs, masked_graphs)):
                masked_cons_idx = masked_graph.masked_cons_idx.cuda()
                graph.x_constraints[masked_cons_idx] = conss[bid].view(-1, 1)

                seen_edges = torch.where(
                    graph.edge_index[0] != masked_cons_idx)[0]
                new_edge_index = torch.LongTensor(
                    [[masked_cons_idx, i] for i in weights_idx[bid]]).T.cuda()
                new_edge_attr = weights[weights_ptr: weights_ptr +
                                        new_edge_index.shape[1]]
                edge_index = torch.cat(
                    (graph.edge_index[:, seen_edges], new_edge_index), dim=-1)
                edge_attr = torch.cat(
                    (graph.edge_attr[seen_edges], new_edge_attr), dim=0)
                graph.edge_index, graph.edge_attr = sort_edge_index(
                    edge_index, edge_attr)

                graphs[bid] = graph
                weights_ptr += len(new_edge_index)

        results = []
        for graph in graphs:
            x_constraints = graph.x_constraints.detach().cpu().numpy()
            edge_index = graph.edge_index.detach().cpu().numpy()
            edge_attr = graph.edge_attr.detach().cpu().numpy()
            x_variables = graph.x_variables.detach().cpu().numpy()
            results.append([x_constraints, edge_index, edge_attr, x_variables])
        return results

    @staticmethod
    def load_model(config: DictConfig, load_model_path: str = None) -> "G2MILP":
        """
        Loads the model.
        """
        data_stats = json.load(open(config.paths.dataset_stats_path, 'r'))
        model = G2MILP(config.model, data_stats).cuda()

        if load_model_path:
            load_ckpt = torch.load(load_model_path)
            model.load_state_dict(load_ckpt, strict=False)

        return model