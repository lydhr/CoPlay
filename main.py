# import matplotlib.pyplot as plt
import os
from itertools import combinations
import hydra
from omegaconf import DictConfig
import logging

import torch
from torch.utils.data import DataLoader

from datasets import *
from models import *
from utils import utils
import utils.dataset_utils as ds_utils
from trainer.trainer import Trainer
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    # ensure CUDA deterministic
    torch.use_deterministic_algorithms(True) 
    torch.backends.cudnn.benchmark = False

utils.init_hydra()

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    utils.set_seed(cfg.seed)

    names = ["pop2019", "pop2022", "podcastConan", "podcastSelena", "piano48"]
    
    #ablation
    # train_ds, test_ds = ds_utils.get_split_datasets(ds_class=YoutubeMixDataset, youtubeDatasetName=names[0], cfg=cfg,  n_test_file=1)
    # train_and_valid(train_ds, test_ds, cfg)

    
    #cross ds
    # train_ds = ds_utils.get_full_dataset(ds_class=BeethovenDataset, cfg=cfg)
    # test_ds = ds_utils.get_full_dataset(ds_class=YoutubeMixDataset, youtubeDatasetName="piano48", cfg=cfg)
    # train_and_test(train_ds, test_ds, cfg)
    # for a, b in [[0,1], [2,3]]:
    #     train_ds = ds_utils.get_full_dataset(ds_class=YoutubeMixDataset, youtubeDatasetName=names[a], cfg=cfg)
    #     test_ds = ds_utils.get_full_dataset(ds_class=YoutubeMixDataset, youtubeDatasetName=names[b], cfg=cfg)
    #     train_and_test(train_ds, test_ds, cfg)

    #plt.show(block=True)

def train_and_test(train_dataset, test_dataset, cfg):
    model = train_valid_random_split(train_dataset, cfg=cfg)
    predict(model, test_dataset, cfg=cfg)

def train_and_valid(train_ds, valid_ds, cfg):
    datasets = {"train": train_ds, "validation": valid_ds}
    return _train_and_valid(datasets, cfg)

def train_valid_random_split(dataset, cfg, valid_size=0.2):
    datasets = ds_utils.random_split_dataset(dataset, test_size=0.2)
    return _train_and_valid(datasets, cfg)

def _train_and_valid(datasets, cfg):
    model = globals()[cfg.model.arch]()
    trainer = Trainer(model=model, datasets=datasets, model_cfg=cfg.model)
    trainer.train()
    return trainer.model

def predict(model, dataset, cfg):
    datasets = {"test": dataset}
    trainer = Trainer(model=model, datasets=datasets, model_cfg=cfg.model)
    trainer.predict()


if __name__ == "__main__":
    main()
