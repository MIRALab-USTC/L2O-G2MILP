import multiprocessing as mp
import os
import os.path as path
from functools import partial
from typing import Union

import numpy as np
from scipy.stats import entropy
from tqdm import tqdm

from src.utils import instance2graph, solve_instance

FEATURES = {
    "coef_dens": float,
    "var_degree_mean": float,
    "var_degree_std": float,
    "cons_degree_mean": float,
    "cons_degree_std": float,
    "lhs_mean": float,
    "lhs_std": float,
    "rhs_mean": float,
    "rhs_std": float,

    "clustering": float,
    "modularity": float,
}


def js_div(x1: np.ndarray, x2: np.ndarray) -> float:
    x = np.hstack([x1, x2])
    if x.std() < 1e-10:
        return 0.0
    M, bins = np.histogram(x, bins=5, density=True)

    P, _ = np.histogram(x1, bins=bins)
    Q, _ = np.histogram(x2, bins=bins)

    return (entropy(P, M) + entropy(Q, M)) / 2


def compute_jsdiv(features1: Union[np.ndarray, str], features2: Union[np.ndarray, str], num_samples: int = 1000):

    f1 = features1[np.random.choice(
        list(range(len(features1))), num_samples, replace=True), :]
    f2 = features2[np.random.choice(
        list(range(len(features2))), num_samples, replace=True), :]

    meta_results = {}
    for i in range(len(FEATURES)):
        jsdiv = js_div(f1[:, i], f2[:, i])
        meta_results[list(FEATURES.keys())[i]] = round(
            1 - jsdiv / np.log(2), 3)

    score = sum(meta_results.values()) / len(meta_results)
    return score, meta_results


def compute_features(samples_dir: str, num_workers: int = 1):
    samples = os.listdir(samples_dir)

    func = partial(compute_features_, data_dir=samples_dir)
    with mp.Pool(num_workers) as pool:
        features = list(tqdm(pool.imap(func, samples), total=len(
            samples), desc="Computing features"))

    return features


def compute_features_(file: str, data_dir: str):
    sample_path = path.join(data_dir, file)
    _, features = instance2graph(sample_path, compute_features=True)
    return features


def solve_instances(samples_dir: str, num_workers: int):
    samples = os.listdir(samples_dir)

    func = partial(solve_instance_, samples_dir=samples_dir)
    with mp.Pool(num_workers) as pool:
        solving_results = list(
            tqdm(pool.imap(func, samples), total=len(samples), desc="Solving instances"))

    return solving_results


def solve_instance_(instance: str, samples_dir: str):
    sample_path = os.path.join(samples_dir, instance)
    solving_results = {"instance": instance}
    solving_results.update(solve_instance(sample_path))
    return solving_results
