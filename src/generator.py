import logging
import os
import os.path as path
from typing import Iterator, List

import torch
from omegaconf import DictConfig
from torch.utils.data import BatchSampler, DataLoader, RandomSampler

from src.data import BipartiteGraph, InstanceDataset
from src.model import G2MILP
from src.utils import graph2instance


class Generator():
    def __init__(self,
                 model: G2MILP,
                 config: DictConfig,
                 templates_dir: str,
                 save_dir: str,
                 ):
        """
        Generator wrapper for G2MILP.

        Args:
            model: G2MILP model
            config: generate config
            templates_dir: directory of template instances
            save_dir: directory to save generated instances
        """
        self.model = model
        self.templates_dir = templates_dir
        self.config = config
        self.samples_dir = save_dir

    def generate(self):
        os.makedirs(self.samples_dir, exist_ok=True)
        template_dataset = InstanceDataset(self.templates_dir)
        template_loader: Iterator[List[BipartiteGraph]] = DataLoader(
            dataset=template_dataset,
            batch_sampler=BatchSampler(
                sampler=RandomSampler(
                    template_dataset, replacement=True, num_samples=self.config.num_samples),
                batch_size=self.config.batch_size,
                drop_last=False),
            collate_fn=lambda x: [x_.cuda() for x_ in x]
        )

        i = 0
        self.model.eval()
        self.model.zero_grad()

        logging.info(
            "="*10 + f"Generating {self.config.num_samples} instances" + "="*10)
        with torch.no_grad():
            for batch in template_loader:
                sample_graphs = self.model.decode(batch, self.config)
                for sample_graph in sample_graphs:
                    i += 1
                    sample_model = graph2instance(sample_graph)
                    sample_path = path.join(
                        self.samples_dir, f"instance_{i}.lp")
                    sample_model.writeProblem(sample_path)
        logging.info("="*40)
