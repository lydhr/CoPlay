import numpy as np
import os
import logging

from utils import file_utils
from .base_dataset import BaseDataset

# A logger for this file
log = logging.getLogger(__name__)

class BeethovenDataset(BaseDataset):
    CONFIG_NAME = "beethoven"
    # DATA_FORMAT = "wav"
    # N_CHANNEL = 1
    
    def load_music(self, files_path, size):
        y = []
        stats = {key: [] for key in self.STATS_KEYS}
        data_list = file_utils.read_wav(files_path)
        for fs, data, data_min, data_max  in data_list:
            y.append(data) # data = (n_sample, n_channel) or (n_sample, ) when n_channel = 1
            self._update_stats(fs, data, data_min, data_max, stats)
        self._log_stats(stats)
        y = np.concatenate(y)
        n = y.shape[0]//size
        return y[:size*n].reshape(n, size)


    def get_files_path(data_cfg, keyword=".wav"):
        """Return files path sorted by the number in the filename. """
        path = data_cfg["datasets"][BeethovenDataset.CONFIG_NAME]["folder_path"] # Get hydra config.
        files = [os.path.join(path, f) for f in os.listdir(path) if keyword in f]
        def get_num(path):
            res = file_utils.get_basename_wo_extension(path)
            res = res.replace('BeethovenPianoSonata', '')\
                    .replace('BeethovenSonata', '')
            return int(res)
        files.sort(key=get_num)
        log.info('{} files: {}...{}'.format(len(files), files[0], files[-1]))
        return files
