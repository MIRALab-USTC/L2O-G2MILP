import json
import logging
import multiprocessing as mp
import os
import os.path as path
import pickle
from functools import partial
from typing import List

import hydra
import numpy as np
from omegaconf import DictConfig
from pandas import DataFrame
from tqdm import tqdm

from src import instance2graph, set_cpu_num, set_seed, solve_instance


def preprocess_(file: str, config: DictConfig):
    """
    Preprocesses a single instance.

    Args:
        file: instance file name
        config: config
    """
    sample_path = path.join(config.paths.data_dir, file)
    data, features = instance2graph(sample_path, config.compute_features)

    with open(path.join(config.paths.dataset_samples_dir, os.path.splitext(file)[0]+".pkl"), "wb") as f:
        pickle.dump(data, f)
    if config.solve_instances:
        solving_results = {"instance": file}
        solving_results.update(solve_instance(sample_path))
    else:
        solving_results = None
    return features, solving_results

def make_dataset_features(
        features: List[dict],
        solving_results: List[dict],
        config:DictConfig
    ):
    """
    Computes the dataset features.

    Args:
        features: list of instance features
        solving_results: list of solving results
        config: config
    """
    if config.compute_features:
        logging.info(f"Writing instance features to: {config.paths.dataset_features_path}")
        features: DataFrame = DataFrame(features, columns=features[0].keys()).set_index("instance")
        features.to_csv(config.paths.dataset_features_path)

        logging.info(f"Writing dataset statistics to: {config.paths.dataset_stats_path}")
        stats = {
            "rhs_type": config.dataset.rhs_type,
            "rhs_min": np.min(features["rhs_min"]),
            "rhs_max": np.max(features["rhs_max"]),

            "obj_type": config.dataset.obj_type,
            "obj_min": np.min(features["obj_min"]),
            "obj_max": np.max(features["obj_max"]),
            
            "coef_type": config.dataset.lhs_type,
            "coef_min": np.min(features['lhs_min']),
            "coef_max": np.max(features["lhs_max"]),
            "coef_dens": np.mean(features["coef_dens"]),

            "cons_degree_min": int(np.min(features["cons_degree_min"])),
            "cons_degree_max": int(np.max(features["cons_degree_max"])),
        }
        with open(config.paths.dataset_stats_path, "w") as f:
            f.write(json.dumps(stats, indent=2))

    if config.solve_instances:
        logging.info(f"Writting solving results to: {config.paths.dataset_solving_path}")
        solving_results: DataFrame = DataFrame(solving_results, columns=solving_results[0].keys()).set_index("instance")
        solving_results.to_csv(config.paths.dataset_solving_path)

        solving_time = solving_results.loc[:, ["solving_time"]].to_numpy()
        num_nodes = solving_results.loc[:, ["num_nodes"]].to_numpy()
        
        logging.info(f"  mean solving time: {solving_time.mean()}")
        logging.info(f"  mean num nodes: {num_nodes.mean()}")

    
@hydra.main(version_base=None, config_path="conf", config_name="preprocess")
def preprocess(config: DictConfig):
    """
    Preprocesses the dataset.
    """
    set_seed(config.seed)
    set_cpu_num(config.num_workers + 1)

    logging.info("="*10 + "Begin preprocessing" + "="*10)
    logging.info(f"Dataset: {config.dataset.name}.")
    logging.info(f"Dataset dir: {config.paths.data_dir}")

    os.makedirs(config.paths.dataset_samples_dir, exist_ok=True)
    os.makedirs(config.paths.dataset_stats_dir, exist_ok=True)

    files: list = os.listdir(config.paths.data_dir)
    files.sort()
    if len(files) > config.dataset.num_train:
        files = files[:config.dataset.num_train]

    func = partial(preprocess_, config=config)
    with mp.Pool(config.num_workers) as pool:
        features, solving_results = zip(*list(tqdm(pool.imap(func, files), total=len(files), desc="Preprocessing")))
    logging.info(f"Preprocessed samples are saved to: {config.paths.dataset_samples_dir}")

    make_dataset_features(features, solving_results, config)
    
    logging.info("="*10 + "Preprocessing finished" + "="*10)


if __name__ == "__main__":
    preprocess()
