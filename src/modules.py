import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import src.tb_writter as tb_writter
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import unbatch
from src.utils import downsample
from torch import Tensor

class ReSample(nn.Module):
    def __init__(self, config) -> None:
        """
        The resample layer in VAE.
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.embd_size = config.embd_size

        self.z_conss_mean = nn.Sequential(
            nn.Linear(self.embd_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.embd_size),
        )
        self.z_vars_mean = nn.Sequential(
            nn.Linear(self.embd_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.embd_size),
        )
      
        self.z_conss_logstd = nn.Sequential(
            nn.Linear(self.embd_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.embd_size),
        )
        self.z_vars_logstd = nn.Sequential(
            nn.Linear(self.embd_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.embd_size),
        )
    
    def forward(self,
            z_conss: Tensor,
            z_vars: Tensor
        ):
        """
        Input:
            z_conss: (num_cons, embd_size), constraint embeddings
            z_vars: (num_vars, embd_size), variable embeddings

        Output:
            z_conss: (num_cons, embd_size), 
            z_vars: (num_vars, embd_size)
            conss_kl_loss: scalar
            vars_kl_loss: scalar
        """
        z_conss_mean = self.z_conss_mean(z_conss)
        z_conss_logstd = torch.clamp_max(self.z_conss_logstd(z_conss), max=10)
        conss_kl_loss = - 0.5 * torch.sum(1.0 + z_conss_logstd - z_conss_mean * z_conss_mean - torch.exp(z_conss_logstd))
        if self.training:
            conss_epsilon = torch.randn_like(z_conss_mean, device=z_conss.device)
            z_conss = z_conss_mean + torch.exp(z_conss_logstd / 2) * conss_epsilon
        else:
            z_conss = z_conss_mean

        z_vars_mean = self.z_vars_mean(z_vars)
        z_vars_logstd = torch.clamp_max(self.z_vars_logstd(z_vars), max=10)
        vars_kl_loss = - 0.5 * torch.sum(1.0 + z_vars_logstd - z_vars_mean * z_vars_mean - torch.exp(z_vars_logstd))
        if self.training:
            vars_epsilon = torch.randn_like(z_vars_mean, device=z_vars.device)
            z_vars = z_vars_mean + torch.exp(z_vars_logstd / 2) * vars_epsilon
        else:
            z_vars = z_vars_mean

        tb = tb_writter.tb_writter
        if self.training:
            tb.add_scalar("resample/conss_kl_loss", conss_kl_loss, tb_writter.step)
            tb.add_scalar("resample/vars_kl_loss", vars_kl_loss, tb_writter.step)
        else:
            tb.add_scalar("resample/conss_kl_loss_val", conss_kl_loss, tb_writter.step)
            tb.add_scalar("resample/vars_kl_loss_val", vars_kl_loss, tb_writter.step)

        return z_conss, z_vars, conss_kl_loss, vars_kl_loss

class Obj_Predictor(nn.Module):
    def __init__(self, config, data_stats):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.embd_size = config.embd_size

        self.obj_type = data_stats["obj_type"]
        self.hidden_size = self.hidden_size

        self.obj_min = int(data_stats["obj_min"])
        self.obj_max = int(data_stats["obj_max"])

        self.var_obj_predictor = nn.Sequential(
            nn.Linear(2 * self.embd_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid(),
        )
       
        self.var_obj_loss = MSELoss(reduction="sum")
    
    def convert(self, obj: torch.Tensor):
        if self.obj_type == "int":
            return torch.round(obj * (self.obj_max - self.obj_min) + self.obj_min)
        elif self.obj_type == "float":
            return obj * (self.obj_max - self.obj_min) + self.obj_min

    def forward(self, z_vars: torch.Tensor, h_vars: torch.Tensor, masked_var_idx: torch.LongTensor, obj_label: torch.Tensor):
        inputs = torch.cat((z_vars[masked_var_idx], h_vars[masked_var_idx]), dim=-1)
        obj_pred = self.var_obj_predictor(inputs).view(-1)

        if abs(self.obj_max - self.obj_min) < 1e-6:
            obj_loss = torch.tensor(0.0, device=inputs.device)
        else:
            obj_label_ = (obj_label.view(-1) - self.obj_min) / (self.obj_max - self.obj_min)
            obj_loss = self.var_obj_loss(obj_pred, obj_label_)
    
        tb = tb_writter.tb_writter
        if self.training:
            tb.add_scalar("Variable_predictor/obj_loss", obj_loss, tb_writter.step)
            tb.add_histogram("Variable_predictor/obj", obj_label, tb_writter.step)
            tb.add_histogram("Variable_predictor/obj_pred", self.convert(obj_pred), tb_writter.step)
        else:
            tb.add_scalar("Variable_predictor/obj_loss_val", obj_loss, tb_writter.step)
            tb.add_histogram("Variable_predictor/obj_val", obj_label, tb_writter.step)
            tb.add_histogram("Variable_predictor/obj_pred_val", self.convert(obj_pred), tb_writter.step)

        return obj_loss, obj_pred

    def decode(self, z_vars: torch.Tensor, h_vars: torch.Tensor, masked_var_idx: torch.LongTensor):
        inputs = torch.cat((z_vars[masked_var_idx], h_vars[masked_var_idx]), dim=-1)
        obj_pred = self.var_obj_predictor(inputs).view(-1)
        return self.convert(obj_pred)

class Type_Predictor(nn.Module):
    def __init__(self, config, data_stats):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.embd_size = config.embd_size

        self.var_type_predictor = nn.Sequential(
            nn.Linear(2 * self.embd_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 4)
        )

        self.var_type_loss = CrossEntropyLoss(reduction="sum")
    
    def convert(self, type: torch.Tensor):
        type = torch.argmax(type, dim=1)
        return torch.zeros((len(type), 4), device=type.device).scatter_(1, type.view(-1,1), 1)

    def forward(self, z_vars: torch.Tensor, h_vars: torch.Tensor, masked_var_idx: torch.LongTensor, type_label: torch.LongTensor):
        inputs = torch.cat((z_vars[masked_var_idx], h_vars[masked_var_idx]), dim=-1)
    
        type_pred = self.var_type_predictor(inputs)
        type_loss = self.var_type_loss(type_pred, type_label.view(-1))

        tb = tb_writter.tb_writter
        if self.training:
            tb.add_scalar("Variable_predictor/type_loss", type_loss, tb_writter.step)
            tb.add_histogram("Variable_predictor/type", type_label, tb_writter.step)
            tb.add_histogram("Variable_predictor/type_pred", self.convert(type_pred), tb_writter.step)
        else:
            tb.add_histogram("Variable_predictor/type_val", type_label, tb_writter.step)
            tb.add_histogram("Variable_predictor/type_pred_val", self.convert(type_pred), tb_writter.step)

        return type_loss, type_pred

    def decode(self, z_vars: torch.Tensor, h_vars: torch.Tensor, masked_var_idx: torch.LongTensor):
        inputs = torch.cat((z_vars[masked_var_idx], h_vars[masked_var_idx]), dim=-1)
        cons_pred = self.var_type_predictor(inputs)
        return self.convert(cons_pred)

class Bias_Predictor(nn.Module):
    def __init__(self, config, data_stats):
        """
        Bias predictor.
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.embd_size = config.embd_size

        self.rhs_type = data_stats["rhs_type"]

        self.rhs_min = int(data_stats["rhs_min"])
        self.rhs_max = int(data_stats["rhs_max"])

        self.cons_predictor = nn.Sequential(
            nn.Linear(2 * self.embd_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid(),
        )

        self.cons_loss = MSELoss(reduction="sum")
    
    def convert(self, cons: torch.Tensor):
        if self.rhs_type == "int":
            return torch.round(cons * (self.rhs_max - self.rhs_min) + self.rhs_min)
        elif self.rhs_type == "float":
            return cons * (self.rhs_max - self.rhs_min) + self.rhs_min

    def forward(self,
            z_conss: torch.Tensor,
            h_conss: torch.Tensor,
            masked_cons_idx: torch.LongTensor,
            cons_label: torch.Tensor
        ):
        inputs = torch.cat((z_conss[masked_cons_idx], h_conss[masked_cons_idx]), dim=-1)
        cons_pred = self.cons_predictor(inputs).view(-1)
        if abs(self.rhs_max - self.rhs_min) < 1e-6:
            cons_loss = torch.tensor(0.0, device=inputs.device)
        else:
            cons_label_ = (cons_label.view(-1) - self.rhs_min) / (self.rhs_max - self.rhs_min)
            cons_loss = self.cons_loss(cons_pred, cons_label_)

        tb = tb_writter.tb_writter
        if self.training:
            tb.add_scalar("Constraint_predictor/cons_loss", cons_loss, tb_writter.step)
            tb.add_histogram("Constraint_predictor/cons", cons_label, tb_writter.step)
            tb.add_histogram("Constraint_predictor/cons_pred", self.convert(cons_pred), tb_writter.step)
        else:
            tb.add_scalar("Constraint_predictor/cons_loss_val", cons_loss, tb_writter.step)
            tb.add_histogram("Constraint_predictor/cons_val", cons_label, tb_writter.step)
            tb.add_histogram("Constraint_predictor/cons_pred_val", self.convert(cons_pred), tb_writter.step)

        return cons_loss, cons_pred

    def decode(self,  z_conss: torch.Tensor, h_conss: torch.Tensor, masked_cons_idx: torch.LongTensor):
        inputs = torch.cat((z_conss[masked_cons_idx], h_conss[masked_cons_idx]), dim=-1)
        cons_pred = self.cons_predictor(inputs).view(-1)
        return self.convert(cons_pred)

class Degree_Predictor(nn.Module):

    def __init__(self, config, data_stats):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.embd_size = config.embd_size

        self.degree_min = data_stats["cons_degree_min"]
        self.degree_max = data_stats["cons_degree_max"]

        self.degree_predictor =  nn.Sequential(
            nn.Linear(2 * self.embd_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        )

        self.degree_loss = MSELoss(reduction="sum")
        
    def convert(self, degree: torch.Tensor):
        return torch.round(degree * (self.degree_max - self.degree_min) + self.degree_min)
    
    def forward(self, z_conss: torch.Tensor, h_conss: torch.Tensor, masked_cons_idx: torch.LongTensor, degree_label: torch.Tensor):
        inputs = torch.cat((z_conss[masked_cons_idx], h_conss[masked_cons_idx]), dim=-1)
        degree_pred = self.degree_predictor(inputs).view(-1)
        degree_label_ = (degree_label - self.degree_min) / (self.degree_max - self.degree_min)
        degree_loss = self.degree_loss(degree_pred, degree_label_)

        tb = tb_writter.tb_writter
        if self.training:
            tb.add_scalar("Degree_predictor/degree_loss", degree_loss, tb_writter.step)
            tb.add_histogram("Degree_predictor/degree", degree_label, tb_writter.step)
            tb.add_histogram("Degree_predictor/degree_pred", self.convert(degree_pred), tb_writter.step)
            tb.add_histogram("Embeddings/h_masked_cons", h_conss[masked_cons_idx], tb_writter.step)
        else:
            tb.add_scalar("Degree_predictor/degree_loss_val", degree_loss, tb_writter.step)
            tb.add_histogram("Degree_predictor/degree_val", degree_label, tb_writter.step)
            tb.add_histogram("Degree_predictor/degree_pred_val", self.convert(degree_pred), tb_writter.step)
            tb.add_histogram("Embeddings/h_masked_cons_val", h_conss[masked_cons_idx], tb_writter.step)

        return degree_loss, degree_pred
    
    def decode(self, z_conss: torch.Tensor, h_conss: torch.Tensor, masked_cons_idx: torch.LongTensor):
        inputs = torch.cat((z_conss[masked_cons_idx], h_conss[masked_cons_idx]), dim=-1)
        degree_pred = self.degree_predictor(inputs).view(-1)
        return self.convert(degree_pred)

class Logits_Predictor(nn.Module):

    def __init__(self, config, data_stats):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.embd_size = config.embd_size

        self.logits_predictor = nn.Sequential(
            nn.Linear(2 * self.embd_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        )

        self.logits_loss = BCEWithLogitsLoss(reduction='sum')

    def forward(self, z_vars: torch.Tensor, h_vars: torch.Tensor, logits_label: torch.Tensor):
       
        logits_input = torch.cat([z_vars, h_vars], dim=-1)
        logits_input, logits_label = downsample(logits_input, logits_label)
        logits_pred = self.logits_predictor(logits_input).view(-1)
        logits_loss = self.logits_loss(logits_pred, logits_label.float())

        tb = tb_writter.tb_writter
        if self.training:
            tb.add_scalar("Logits_predictor/logits_loss", logits_loss, tb_writter.step)

            logits_label = logits_label.cpu().numpy()
            logits_pred = logits_pred.detach().cpu().numpy()
            auc = roc_auc_score(logits_label, logits_pred)
            tb.add_scalar("Logits_predictor/logits_auc", auc, tb_writter.step)
            logits_pred = (logits_pred > 0)
            tb.add_histogram("Logits_predictor/logits", logits_label, tb_writter.step)
            tb.add_histogram("Logits_predictor/logits_pred", logits_pred, tb_writter.step)

        else:
            tb.add_scalar("Logits_predictor/logits_loss_val", logits_loss, tb_writter.step)

            logits_label = logits_label.cpu().numpy()
            logits_pred = logits_pred.detach().cpu().numpy()
            auc = roc_auc_score(logits_label, logits_pred)
            tb.add_scalar("Logits_predictor/logits_auc_val", auc, tb_writter.step)
            logits_pred = (logits_pred > 0)
            tb.add_histogram("Logits_predictor/logits_val", logits_label, tb_writter.step)
            tb.add_histogram("Logits_predictor/logits_pred_val", logits_pred, tb_writter.step)

        return logits_loss, logits_pred
        
    
    def decode(self, z_vars: torch.Tensor, h_vars: torch.Tensor, x_variables_batch: torch.Tensor, degree: torch.LongTensor):
        logits_input = torch.cat([z_vars, h_vars], dim=-1)
        logits_pred = self.logits_predictor(logits_input).view(-1)
        logits_pred = unbatch(logits_pred, x_variables_batch)

        logits = [logits_pred[i].view(-1) for i in range(len(logits_pred))]

        non_zeors = []
        for degree_, logits_ in zip(degree, logits):
            degree__ = torch.minimum(degree_, torch.tensor(logits_.shape[0]))
            _, indices = torch.topk(logits_, int(degree__.item()))
            non_zeors.append(indices)
            
        return non_zeors

class Weights_Predictor(nn.Module):

    def __init__(self, config, data_stats):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.embd_size = config.embd_size

        self.weights_type = data_stats["coef_type"]
        self.weights_min = int(data_stats["coef_min"])
        self.weights_max = int(data_stats["coef_max"])

        self.weights_predictor = nn.Sequential(
            nn.Linear(2 * self.embd_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        )

        self.weights_loss = MSELoss(reduction="sum")

    def convert(self, weights: torch.Tensor):
        if self.weights_type == "int":
            return torch.round(weights * (self.weights_max - self.weights_min) + self.weights_min)
        else:
            return weights * (self.weights_max - self.weights_min) + self.weights_min
        
    def forward(self, z_vars, h_vars, weights_idx, weights_label):
        weights_input = torch.cat((z_vars[weights_idx], h_vars[weights_idx]), dim=-1)
        weights_pred = self.weights_predictor(weights_input).view(-1)
        if abs(self.weights_max - self.weights_min) < 1e-6:
            weights_loss = torch.tensor(0.0).to(weights_pred.device)
        else:
            weights_label_ = (weights_label.view(-1) - self.weights_min) / (self.weights_max - self.weights_min)
            weights_loss = self.weights_loss(weights_pred, weights_label_)

        tb = tb_writter.tb_writter
        if self.training:
            tb.add_scalar("Weights_predictor/weights_loss", weights_loss, tb_writter.step)
            tb.add_histogram("Weights_predictor/weights", weights_label, tb_writter.step)
            tb.add_histogram("Weights_predictor/weights_pred", self.convert(weights_pred), tb_writter.step)
        else:
            tb.add_scalar("Weights_predictor/weights_loss_val", weights_loss, tb_writter.step)
            tb.add_histogram("Weights_predictor/weights_val", weights_label, tb_writter.step)
            tb.add_histogram("Weights_predictor/weights_pred_val", self.convert(weights_pred), tb_writter.step)
        return weights_loss, weights_pred

    def decode(self, z_vars, h_vars, x_variables_batch, weights_idx):
        weights_input = []
        for h_vars_, z_, weights_idx_ in zip(unbatch(h_vars, x_variables_batch), unbatch(z_vars, x_variables_batch), weights_idx):
            weights_input.append(torch.cat((h_vars_[weights_idx_], z_[weights_idx_]), dim=-1))
        weights_input = torch.cat(weights_input, dim=0)
        weights = self.weights_predictor(weights_input)
        return self.convert(weights)