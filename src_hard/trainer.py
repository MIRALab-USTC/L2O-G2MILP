import pandas as pd
import logging
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from omegaconf import DictConfig
import sys
import json
import os
from typing import Iterator
from src_hard.data import InstanceDataset
sys.path.append(os.getcwd())
from src.model import G2MILP
from src_hard.generator import Generator
from src.data import BipartiteGraph, TrainSampler
import src.tb_writter as tb_writter
from src.trainer import beta_scheduler

import torch
import torch.optim as optim
from typing import Union, List, Dict
import os.path as path
from src.benchmarks.utils import solve_instances
from src.utils import instance2graph
import pickle
from functools import partial
import multiprocessing as mp
from tqdm import tqdm

class Storage:
    def __init__(self,
            init_data_dir: str,
            init_solving_path: str,
            init_samples_dir: str,
            capacity: int,
            ) -> None:
        """
        The storage of instances.

        Args:
            init_data_dir: dir for the initial instances
            init_solving_path: path to the solving results of the initial instances
            init_samples_dir: dir for the preprocessed initial samples
            capacity (int): the maximum number of instances to store
        """
        solving_results = pd.read_csv(init_solving_path).set_index("instance")
        self.instance_list = [
            {
            'name': os.path.splitext(f)[0],
            'instance_path': os.path.join(init_data_dir, f),
            'sample_path': os.path.join(init_samples_dir, os.path.splitext(f)[0]+".pkl"),
            'solving_status': int(solving_results["status"][f]),
            'solving_time': float(solving_results["solving_time"][f])}
            for f in os.listdir(init_data_dir)
        ]
        self.capacity = capacity
        self.update_storage()

    def save_storage(self, dir: str):
        os.makedirs(dir, exist_ok=True)
        with open(os.path.join(dir, "storage.json"), "w") as f:
            json.dump(self.instance_list, f, indent=4)
    
    def len(self):
        return len(self.instance_list)
    
    def get_train_set(self):
        return InstanceDataset(self.instance_list)

    def extend(self, new_instances: List[Dict]):
        self.instance_list.extend(new_instances)
        self.update_storage()

    def update_storage(self):
        self.instance_list = [instance for instance in self.instance_list if instance['solving_status'] == 2]
        self.instance_list = sorted(self.instance_list, key=lambda x: x['solving_time'], reverse=True)[:self.capacity]

class Trainer:
    def __init__(self,
                 model: G2MILP,
                 train_set: InstanceDataset,
                 loop: int,
                 workspace: str,
                 config: DictConfig,
                 generator_config: DictConfig,
                 ):
        """
        Trainer for G2MILP in one loop.

        Args:
            model: G2MILP model
            train_set: training dataset
            loop: the current loop
            workspace: dir to save the training results
            config: training config
            generator_config: generator config
        """
        self.model = model
        self.config = config
        self.loop = loop
        
        # laod paths
        self.workspace = workspace
        os.makedirs(self.workspace, exist_ok=True)
        self.model_dir = os.path.join(self.workspace, "model")
        os.makedirs(self.model_dir, exist_ok=True)

        # load parameters
        self.total_steps = config.steps
        self.step = 0
        self.save_step = config.save_step

        self.grad_max_norm = config.grad_max_norm

        # prepare the data loader
        self.batch_size = config.batch_size
        self.batch_repeat_size = config.batch_repeat_size

        self.train_set = train_set
        self.train_loader: Iterator[BipartiteGraph] = DataLoader(
            dataset=train_set,
            batch_sampler=TrainSampler(
                train_set,
                batch_size=self.batch_size,
                repeat_size=self.batch_repeat_size,
                total_steps=self.total_steps,
            ),
            follow_batch=["x_constraints", "x_variables"],
        )
    
        # init optimizer
        self.optimizer = optim.Adam(
            model.parameters(), lr=config.lr.init, weight_decay=config.weight_decay)

        # init lr scheduler
        self.lr_anneal_step = config.lr.anneal_step
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer, config.lr.anneal_rate)

        # init beta scheduler
        self.beta_cons, self.beta_var = config.beta.cons.min, config.beta.var.min
        self.beta_cons_scheduler = beta_scheduler(config.beta.cons)
        self.beta_var_scheduler = beta_scheduler(config.beta.var)

        self.performance_track = []
        self.save_start = config.save_start

        self.generator_config = generator_config

    def train(self):
        """
        Train the model.
        """
        for data in self.train_loader:
            data = data.cuda()
            self.step += 1
            tb_writter.set_step(self.step)
            self.step_train(data)

            self.beta_cons = self.beta_cons_scheduler.step()
            self.beta_var = self.beta_var_scheduler.step()

            if self.step % self.save_step == 0 and self.step >= self.save_start:
                self.step_save()

            if self.step % self.lr_anneal_step == 0:
                self.lr_scheduler.step()

        self.save_best_ckpt()

    def step_train(self, data: Union[Batch, BipartiteGraph]):
        """
        Train the model for one step.
        """
        self.model.train()
        self.model.zero_grad()
        torch.cuda.empty_cache()
        loss = self.model.forward(
            data, beta_cons=self.beta_cons, beta_var=self.beta_var)
        loss.backward()

        tb_writter.add_scalar("Train/lr", self.lr, self.step)
        tb_writter.add_scalar("Train/beta_cons", self.beta_cons, self.step)
        tb_writter.add_scalar("Train/beta_var", self.beta_var, self.step)
        tb_writter.add_scalar("Train/total_loss", loss, self.step)

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.grad_max_norm)
        self.optimizer.step()

        logging.info(
            f"Step {self.step}/{self.total_steps}. Loss: {loss.item():.5f}. beta_cons: {self.beta_cons:.5f}. beta_var: {self.beta_var:.5f}. lr: {self.lr}.")

    def step_save(self):
        """
        Save the model and generate samples.
        """
        self.model.eval()
        self.model.zero_grad()
        torch.cuda.empty_cache()
        save_path = path.join(self.model_dir, f"model_step_{self.step}.ckpt")
        with torch.no_grad():
            torch.save(self.model.state_dict(), save_path)

        instances_dir = os.path.join(self.workspace, f"steps/step_{self.step}/instances")
        os.makedirs(instances_dir, exist_ok=True)
        generator = Generator(
            model=self.model,
            config=self.generator_config,
            template_dataset=self.train_set,
            save_dir=instances_dir
        )
        generator.generate()

        solving_path = os.path.join(self.workspace, f"steps/step_{self.step}/solving.csv")
        samples_solving_results = solve_instances(instances_dir, self.config.num_workers)
        samples_solving_results = pd.DataFrame(samples_solving_results).set_index("instance")
        samples_solving_results.to_csv(solving_path)

        instances_info = [
            {
            'name': f"loop{self.loop}_step{self.step}_{os.path.splitext(f)[0]}",
            'instance_path': os.path.join(instances_dir, f),
            'sample_path': None,
            'solving_status': int(samples_solving_results["status"][f]),
            'solving_time': float(samples_solving_results["solving_time"][f])}
            for f in os.listdir(instances_dir)
        ]
        with open(os.path.join(self.workspace, f"steps/step_{self.step}/instances_info.json"), "w") as f:
            json.dump(instances_info, f, indent=4)

        mean_solving_time = samples_solving_results["solving_time"].mean()
        max_solving_time = samples_solving_results["solving_time"].max()
        tb_writter.add_scalar("Train/solving_time", mean_solving_time, self.step)
        tb_writter.add_scalar("Train/solving_time_max", max_solving_time, self.step)
        self.performance_track.append((self.step, mean_solving_time))
       
    def save_best_ckpt(self):
        """
        Save the best model.
        """
        self.performance_track.sort(key=lambda x: x[1], reverse=True)
        best_step = self.performance_track[0][0]

        logging.info(f"Best step: {best_step}.")
        best_model_path = path.join(
            self.model_dir, f"model_step_{best_step}.ckpt")
        self.model.eval()
        self.model.zero_grad()
        self.model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cuda')))
        torch.cuda.empty_cache()
        save_path = path.join(self.model_dir, f"model_best.ckpt")
        with torch.no_grad():
            torch.save(self.model.state_dict(), save_path)
        logging.info(f"Best model saved to {save_path}.")

    @property
    def lr(self):
        return self.lr_scheduler.get_last_lr()[0]

def preprocess_(instance: dict, samples_dir: str):
    instance_path = instance['instance_path']
    sample_path = os.path.join(samples_dir, instance['name'] + ".pkl")
    data, _ = instance2graph(instance_path, compute_features=False)
    with open(sample_path, "wb") as f:
        pickle.dump(data, f)

class HardTrainer:
    def __init__(self,
                 model: G2MILP,
                 init_data_dir: str,
                 init_solving_path: str,
                 init_samples_dir: str,
                 workspace: str,
                 config: DictConfig,
                 generator_config: DictConfig,
                 ) -> None:
        """
        Trainer for G2MILP to generate hard instances.

        Args:
            model: G2MILP model
            init_data_dir: dir for the initial instances
            init_solving_path: path to the solving results of the initial instances
            init_samples_dir: dir for the preprocessed initial samples
            workspace: dir to save the training results
            config: training config
            generator_config: generator config
        """
        self.model = model
        self.storage = Storage(
            init_data_dir=init_data_dir,
            init_solving_path=init_solving_path,
            init_samples_dir=init_samples_dir,
            capacity=config.storage_capacity,
        )

        self.workspace = workspace

        self.trainer_config = config
        self.generator_config = generator_config

    def generate_and_update_storage(self, loop: int):
        """
        Generate samples and update the storage in one loop.
        """
        self.generate(loop)

        new_instances = []
        workspace = os.path.join(self.workspace, f"loop{loop}")
        for step_dir in os.listdir(os.path.join(workspace, "steps")):
            json_path = os.path.join(workspace, "steps", step_dir, "instances_info.json")
            new_instances.extend(json.load(open(json_path)))
        new_instances.extend(json.load(open(os.path.join(workspace, "best_step", "instances_info.json"))))
        
        self.storage.extend(new_instances)
    
    def preprocess_loop(self, loop: int):
        """
        Preprocess the instances in one loop.
        """
        not_processed_instances = [instance for instance in self.storage.instance_list if instance['sample_path'] == None]
        if len(not_processed_instances) > 0:
            samples_dir = os.path.join(self.workspace, f"loop{loop}/data_samples")
            os.makedirs(samples_dir, exist_ok=True)
            func = partial(preprocess_, samples_dir=samples_dir)
            with mp.Pool(self.trainer_config.num_workers) as pool:
                tqdm(pool.map(func, not_processed_instances), total=len(not_processed_instances), desc="Preprocessing")
            for instance in self.storage.instance_list:
                if instance['sample_path'] == None:
                    instance['sample_path'] = os.path.join(samples_dir, instance['name'] + ".pkl")
                    
    def train_loop(self, loop: int):
        """
        Train the model in one loop.
        """    
        trainer = Trainer(
            model=self.model,
            train_set=self.storage.get_train_set(),
            loop=loop,
            workspace=os.path.join(self.workspace, f"loop{loop}"),
            config=self.trainer_config,
            generator_config=self.generator_config,
        )

        trainer.train()

        self.generate_and_update_storage(loop)

    def generate(self, loop: int):
        best_model_path = os.path.join(self.workspace, f"loop{loop}/model/model_best.ckpt")
        self.model.eval()
        self.model.zero_grad()
        self.model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cuda')))
        torch.cuda.empty_cache()
        
        instances_dir = os.path.join(self.workspace, f"loop{loop}/best_step/instances")
        os.makedirs(instances_dir, exist_ok=True)
        generator_config = self.generator_config.copy()
        generator_config.num_samples = self.generator_config.loop_num_samples
        generator = Generator(
            model=self.model,
            config=generator_config,
            template_dataset=self.storage.get_train_set(),
            save_dir=instances_dir
        )
        generator.generate()

        solving_path = os.path.join(self.workspace, f"loop{loop}/best_step/solving.csv")
        samples_solving_results = solve_instances(instances_dir, self.trainer_config.num_workers)
        samples_solving_results = pd.DataFrame(samples_solving_results).set_index("instance")
        samples_solving_results.to_csv(solving_path)

        instances_info = [
            {
            'name': f"loop{loop}_best_step_{os.path.splitext(f)[0]}",
            'instance_path': os.path.join(instances_dir, f),
            'sample_path': None,
            'solving_status': int(samples_solving_results["status"][f]),
            'solving_time': float(samples_solving_results["solving_time"][f])}
            for f in os.listdir(instances_dir)
        ]
        with open(os.path.join(self.workspace, f"loop{loop}/best_step/instances_info.json"), "w") as f:
            json.dump(instances_info, f, indent=4)




        