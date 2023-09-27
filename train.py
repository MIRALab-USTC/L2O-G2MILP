import logging
import os.path as path

import hydra
import torch
from omegaconf import DictConfig

import src.tb_writter as tb_writter
from src import (G2MILP, Benchmark, Generator, InstanceDataset, Trainer,
                 set_cpu_num, set_seed)


@hydra.main(version_base=None, config_path="conf", config_name="train")
def train(config: DictConfig):
    """
    Train G2MILP.
    """
    set_seed(config.seed)
    set_cpu_num(config.num_workers + 1)
    torch.cuda.set_device(config.cuda)
    tb_writter.set_logger(config.paths.tensorboard_dir)

    model = G2MILP.load_model(config)
    logging.info(f"Loaded model.")
    logging.info(
        f"  Number of model parameters: {sum([x.nelement() for x in model.parameters()]) / 1000}K.")

    train_set = InstanceDataset(
        data_dir=config.paths.dataset_samples_dir,
        solving_results_path=config.paths.dataset_solving_path,
    )
    logging.info(f"Loaded dataset.")
    logging.info(f"  Number of training instances: {len(train_set)}.")
   
    trainer = Trainer(
        model=model,
        train_set=train_set,
        paths=config.paths,
        config=config.trainer,
        generator_config=config.generator,
        benchmark_config=config.benchmarking,
    )

    logging.info("="*10 + "Begin training" + "="*10)

    trainer.train()

    logging.info("="*10 + "Training finished" + "="*10)

    # test
    for mask_ratio in [0.01, 0.05, 0.1]:
        config.generator.mask_ratio = mask_ratio

        # generate
        samples_dir = path.join(config.paths.train_dir,
                                f"eta-{mask_ratio}/samples")
        generator = Generator(
            model=model,
            config=config.generator,
            templates_dir=config.paths.dataset_samples_dir,
            save_dir=samples_dir,
        )
        generator.generate()

        # benchmark
        benchmark_dir = path.join(
            config.paths.train_dir, f"eta-{mask_ratio}/benchmark")
        benchmarker = Benchmark(
            config=config.benchmarking,
            dataset_stats_dir=config.paths.dataset_stats_dir,
        )
        results = benchmarker.assess_samples(
            samples_dir=samples_dir,
            benchmark_dir=benchmark_dir
        )

        info_path = path.join(benchmark_dir, "info.json")
        benchmarker.log_info(
            generator_config=config.generator,
            benchmarking_config=config.benchmarking,
            meta_results=results,
            save_path=info_path,
        )


if __name__ == '__main__':
    train()
