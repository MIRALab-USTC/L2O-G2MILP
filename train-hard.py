import os
import logging
import hydra
import torch
from omegaconf import DictConfig

import src.tb_writter as tb_writter
from src import (G2MILP, set_cpu_num, set_seed)
from src_hard import HardTrainer


@hydra.main(version_base=None, config_path="conf", config_name="train-hard")
def train(config: DictConfig):
    """
    Train G2MILP to generate hard instances.
    """
    set_seed(config.seed)
    set_cpu_num(config.num_workers + 1)
    torch.cuda.set_device(f"cuda:{config.cuda}")
    tb_writter.set_logger(config.paths.tensorboard_dir)

    model = G2MILP.load_model(config, load_model_path=config.pretrained_model_path)
    logging.info(f"Loaded model.")
    logging.info(
        f"  Number of model parameters: {sum([x.nelement() for x in model.parameters()]) / 1000}K.")
    logging.info(
        f"  Load pretrained model from: {config.pretrained_model_path}")

    trainer = HardTrainer(
        model=model,
        init_data_dir=config.paths.data_dir,
        init_solving_path=config.paths.dataset_solving_path,
        init_samples_dir=config.paths.dataset_samples_dir,
        workspace=config.paths.train_dir,
        config=config.trainer_hard,
        generator_config=config.generator,
    )

    logging.info("="*10 + "Begin training" + "="*10)

    for loop in range(config.trainer_hard.num_loops):
        logging.info("="*10 + f"Loop {loop:2d}" + "="*10)
        
        trainer.preprocess_loop(loop)
        trainer.storage.save_storage(os.path.join(config.paths.train_dir, f"loop{loop}"))
        trainer.train_loop(loop)

    trainer.storage.save_storage(config.paths.train_dir)

    logging.info("="*10 + "Training finished" + "="*10)


if __name__ == '__main__':
    train()
