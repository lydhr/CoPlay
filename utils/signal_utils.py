import numpy as np
import logging
import torch
import matplotlib.pyplot as plt
from utils import utils

# A logger for this file
log = logging.getLogger(__name__)

class SignalType:
    sine = "sine"
    FMCW = "FMCW"

def generate_FMCW(f_start, f_end, fs, size, max_amplitude):
    """
        self.simulationData = (n_samples, )
    """
    t = np.arange(0, size,1)/fs
    k = (f_end - f_start)/(size/fs)
    phase = np.pi * 2 * (f_start * t + 0.5 * k * np.power(t, 2)) % (2 * np.pi)
    samples = np.round(np.sin(phase) * max_amplitude)

    log_sim_samples_stats(samples, fs, f"FMCW {f_start}-{f_end}hz")
    return samples

def generate_sine(f, fs, size, max_amplitude):
    t = np.arange(0, size, 1)/fs
    phase = np.pi * 2 * f * t
    samples = np.round(np.sin(phase) * max_amplitude) #18k * 512/48k = 192 periods

    log_sim_samples_stats(samples, fs, f"sine {f}hz")
    return samples

def log_sim_samples_stats(samples, fs, signal_type):
    log.info("generated {}  = {} = {:.3f}s".format(signal_type, samples.shape, samples.shape[0]/fs))

def get_scaled_FFT(y, amplitude):
    '''
        y: numpy or torch.tensor, (n, win_size), or (ws,) 
        amplitude: numpy or torch.tensor, (n,) or ()
            amplitude is not necessarily the m in config.yaml. It depends on y.
            numerical range: z in [-m-1, m], x in [+-m], xhat in [+-m]-[-m-1, m]=[-2m-1, 2m]
        return: torch.tensor, (n, win_size//2+1), or (ws//2+1,)
             win_size//2+1 because it is Hermitian-symmetric, X[i] = conj(X[-i]), e.g. X[1] = X[-1], X[0] is dummy = sum(x_i...)
             so the output contains only the positive frequencies below the Nyquist frequency.
    '''
    freq = get_FFT(y)
    window_size = y.shape[-1]
    if len(amplitude.shape) != 0: #(n,)
        amplitude = amplitude[:, None] #(n, 1)
    freq = freq/(window_size/2)/amplitude #norm the freq to [0-1]
    log.debug("fft coefficients range {:.6f}-{:.6f}".format(torch.min(freq).item(), torch.max(freq).item()))
    return freq

def exclude_column(x, i): 
    '''
        x: (batch_size, window_size)
        return: (batch_size, win_size-1)
    '''
    return torch.cat((x[:, :i], x[:, i+1:]), dim = 1)

def exclude_column_range(x, start, end):
    '''
        x: (batch_size, window_size)
        return: (batch_size, win_size-(end-start+1))
    '''
    return torch.cat((x[:, :start], x[:, end+1:]), dim = 1)

def get_FFT(y):
    '''
        y: torch.tensor or numpy.ndarry, (n, win_size) or (ws, )
        return: torch.tensor (n, win_size//2+1) or (ws//2+1)
    '''
    if type(y) is np.ndarray:
        y = torch.from_numpy(y)
    return torch.abs(torch.fft.rfft(y))

def plot_FFT(x, xhat, m, cfg, save_path=None):
    '''
        x: numpy array, (window_size,), ground truth
        xhat: numpy array, (n, window_size)
        m: numpy array, ()
        save_path: str or None
    '''
    plt.figure()
    plt.ylim(0, cfg.ylim_max)
    # xhat, down sampled
    idx = utils.get_down_sample_idx(xhat, cfg.down_sample)
    xhat = np.array([xhat[i] for i in idx])
    freq = get_scaled_FFT(xhat, m*2+1)
    freq = freq.detach().cpu().numpy()
    for i in range(len(freq)):
        plt.plot(freq[i])
    
    # x, ground truth
    freq = get_scaled_FFT(x, m*2+1)
    freq = freq.detach().cpu().numpy()
    plt.plot(freq, color='orange', marker='x', linestyle='dashed', label='sim')
    
    plt.legend(loc='upper right')
    plt.title('[{}]'.format('/'.join(save_path.split('/')[-3:])))
    # save and show
    if save_path is not None:
        plt.savefig(save_path, bbox_inches = "tight")
        log.debug('Saved image in {}'.format(save_path))
     
    if cfg.show:
        plt.show(block=False)
        plt.pause(0.001)

