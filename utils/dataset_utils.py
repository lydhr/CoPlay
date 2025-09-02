import torch
import logging

log = logging.getLogger(__name__)

def random_split_dataset(dataset, test_size=0.2):
    test_size = int(test_size * len(dataset))
    train_size = len(dataset) - test_size
    datasets = {}
    datasets["train"], datasets["validation"] = torch.utils.data.random_split(dataset, [train_size, test_size])
     
    log.info("random split, {}".format([{key: len(datasets[key])} for key in datasets]))
    return datasets

def get_split_datasets(ds_class, cfg, youtubeDatasetName="", n_test_file=1, keyword=".wav"):
    files_path = get_files_path(ds_class=ds_class, data_cfg=cfg.data, youtubeDatasetName=youtubeDatasetName, keyword=keyword)

    train_ds = ds_class(fnames=files_path[:-n_test_file], signal_cfg=cfg.signal)
    test_ds = ds_class(fnames=files_path[-n_test_file:], signal_cfg=cfg.signal)
    return train_ds, test_ds

def get_full_dataset(ds_class, cfg, youtubeDatasetName="", keyword=".wav"):
    files_path = get_files_path(ds_class=ds_class, data_cfg=cfg.data, youtubeDatasetName=youtubeDatasetName, keyword=keyword)
    ds = ds_class(fnames=files_path, signal_cfg=cfg.signal)
    return ds

def get_files_path(ds_class, data_cfg, youtubeDatasetName, keyword):
    argkv = {'data_cfg': data_cfg, 'keyword': keyword}
    if youtubeDatasetName:
        argkv['dataset'] = youtubeDatasetName
    files_path = ds_class.get_files_path(**argkv)
    return files_path
