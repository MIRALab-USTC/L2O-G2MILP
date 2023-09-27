import json
import logging
import os

import pandas as pd
from omegaconf import DictConfig, OmegaConf


from .utils import FEATURES, compute_features, compute_jsdiv, solve_instances


class Benchmark():
    def __init__(self, config: DictConfig, dataset_stats_dir: str):
        self.config = config
        self.dataset_stats_dir = dataset_stats_dir

    def assess_samples(self,
                       samples_dir: str,
                       benchmark_dir: str
                       ):
        """
        Assess the generated samples.

        Args:
            samples_dir: directory of generated samples
            benchmark_dir: directory to save benchmark results

        Returns:
            results: dict, benchmark results
        """
        os.makedirs(benchmark_dir, exist_ok=True)
        distribution_results = assess_distribution(
            self.config, samples_dir, self.dataset_stats_dir, benchmark_dir)
        solving_results = assess_solving_results(
            self.config, samples_dir, self.dataset_stats_dir, benchmark_dir)
        results = {
            "distribution": distribution_results,
            "solving": solving_results,
        }
        return results

    @staticmethod
    def log_info(
        generator_config: DictConfig,
        benchmarking_config: DictConfig,
        meta_results: dict,
        save_path: str
    ):
        info = {
            "generator": OmegaConf.to_container(generator_config),
            "benchmarking": OmegaConf.to_container(benchmarking_config),
            "meta_results": meta_results,
        }
        logging.info("-"*10 + " Benchmarking  Info " + "-"*10)
        json_msg = json.dumps(info, indent=4)
        for msg in json_msg.split("\n"):
            logging.info(msg)
        logging.info("-"*40)

        if save_path is not None:
            with open(save_path, 'w') as f:
                f.write(json.dumps(info, indent=4))
            logging.info(f"Benchmarking info saved to {save_path} .")


def assess_distribution(config, samples_dir, dataset_stats_dir, benchmark_dir):
    samples_features = compute_features(
        samples_dir, num_workers=config.num_workers)
    samples_features = pd.DataFrame(samples_features).set_index("instance")
    samples_features.to_csv(os.path.join(benchmark_dir, "features.csv"))

    reference_features = pd.read_csv(os.path.join(
        dataset_stats_dir, "features.csv")).set_index("instance")

    used_features = list(FEATURES.keys())
    samples_features = samples_features.loc[:, used_features].to_numpy()
    reference_features = reference_features.loc[:, used_features].to_numpy()

    score, meta_results = compute_jsdiv(
        samples_features, reference_features, num_samples=config.num_samples)

    results = {
        "score": score,
        "meta_results": meta_results,
    }
    return results


def assess_solving_results(config, samples_dir, dataset_stats_dir, benchmark_dir):
    samples_solving_results = solve_instances(
        samples_dir, num_workers=config.num_workers)
    samples_solving_results = pd.DataFrame(
        samples_solving_results).set_index("instance")
    samples_solving_results.to_csv(os.path.join(
        benchmark_dir, "solving_results.csv"))

    reference_solving_results = pd.read_csv(os.path.join(
        dataset_stats_dir, "solving_results.csv")).set_index("instance")

    samples_solving_time = samples_solving_results.loc[:, [
        "solving_time"]].to_numpy()
    samples_num_nodes = samples_solving_results.loc[:, [
        "num_nodes"]].to_numpy()
    reference_solving_time = reference_solving_results.loc[:, [
        "solving_time"]].to_numpy()
    reference_num_nodes = reference_solving_results.loc[:, [
        "num_nodes"]].to_numpy()

    results = {
        "solving_time": {
            "mean": samples_solving_time.mean(),
            "mean_reference": reference_solving_time.mean(),
            "mean_error": abs(samples_solving_time.mean() - reference_solving_time.mean()) / reference_solving_time.mean(),
        },
        "num_nodes": {
            "mean": samples_num_nodes.mean(),
            "mean_reference": reference_num_nodes.mean(),
            "mean_error": abs(samples_num_nodes.mean() - reference_num_nodes.mean()) / reference_num_nodes.mean(),
        },
    }
    return results
