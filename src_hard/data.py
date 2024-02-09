from src.data import BipartiteGraph
import os, sys
import pickle
from torch_geometric.data import Dataset
from torch import FloatTensor, LongTensor
import pandas as pd
from typing import List

class InstanceDataset(Dataset):
    def __init__(self, instance_list: List[dict]):
        super().__init__()
        self.files = [instance["sample_path"] for instance in instance_list]
    
    def len(self):
        return len(self.files)

    def get(self, index):
        file = self.files[index]
        with open(file, "rb") as f:
            data = pickle.load(f)
        x_constraints, edge_index, edge_attr, x_variables = data

        return BipartiteGraph(
            x_constraints=FloatTensor(x_constraints),
            edge_index=LongTensor(edge_index),
            edge_attr=FloatTensor(edge_attr),
            x_variables=FloatTensor(x_variables),
        )