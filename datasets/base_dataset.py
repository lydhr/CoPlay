import numpy as np
import logging
import torch
from torch.utils.data import Dataset

from utils import signal_utils

# A logger for this file
log = logging.getLogger(__name__)

class BaseDataset(Dataset):
    STATS_KEYS = ['t', 'fs', 'min', 'max']

    def __init__(self, fnames, signal_cfg):
        self.parse_config(signal_cfg)
        m1 = signal_cfg.numerical_range
        self.m = m1 * 2 # xhat + z <= 2m
        self.m_z = m1 * signal_cfg.music_m_scaler # z from raw file ~<=m1
        self.m_x = self.m - self.m_z
        self.m_xhat = self.m + self.m_z # -m - z <= xhat <= m - z
        self.x = self.get_sensing(maxi = self.m_x)
        self.z = self.get_music(fnames) * signal_cfg.music_m_scaler
        self.shape() #(512,), (xxx, 512)

    def parse_config(self, cfg):
        self.FS = cfg.fs
        self.WINDOW_SIZE = cfg.window_size
        self.signal_type = cfg.type
        if self.signal_type == "sine":
            self.SINE_F = cfg.sine.frequency
        elif self.signal_type == "FMCW":
            self.F_START = cfg.FMCW.frequency['start']
            self.F_END = cfg.FMCW.frequency['end']

    def get_sensing(self, maxi):
        x = None
        match self.signal_type:
            case signal_utils.SignalType.sine:
                x = signal_utils.generate_sine(
                    f=self.SINE_F, fs=self.FS, 
                    size=self.WINDOW_SIZE, max_amplitude=maxi
                )
            case signal_utils.SignalType.FMCW:
                x = signal_utils.generate_FMCW(
                    f_start=self.F_START, f_end=self.F_END,
                    fs=self.FS, size=self.WINDOW_SIZE, max_amplitude=maxi
                )
            case default:
                raise ValueError(f"invalid SignalType {self.signal_type}")
        return self.set_precision(x)

    def get_music(self, files_path):
        if len(files_path) < 1:
            raise ValueError('Dataset input files_path is empty.')
        z = self.load_music(files_path=files_path, size=self.WINDOW_SIZE)
        return self.set_precision(z)

    def set_precision(self, data, precision=np.float32):
        #otherwise, model.double()
        if type(data) != precision:
            return data.astype(precision)
        else:
            return data

    def shape(self):
        d_shape = (self.x.shape, self.z.shape)
        log.info("{} (x.shape, z.shape) = {}".format(type(self).__name__, d_shape))
        return d_shape
        
    def __getitem__(self, idx):
        return self.x, self.z[idx], self.m

    def __len__(self):
        return len(self.z)
    
    def _update_stats(self, fs, data, data_min, data_max, stats):
        # stats[] outside will be updated
        stats['t'].append(data.shape[0]/fs)
        stats['fs'].append(fs)
        stats['min'].append(data_min)
        stats['max'].append(data_max)
    
    def _log_stats(self, stats):
        for name, arr in stats.items():
            if len(arr) == 0:
                raise ValueError("{} has empty arr".format(name))
            log.info("data stats: {} in [{}, {}]".format(name, min(arr), max(arr)))
        
