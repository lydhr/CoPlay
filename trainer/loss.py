import logging
import torch
from utils import utils, signal_utils

# A logger for this file
log = logging.getLogger(__name__)

def get_loss(x, z, m, xhat, loss_cfg, returnSub=False):
    match loss_cfg.type:
        case signal_utils.SignalType.FMCW:
            return FMCW_loss(x=x, z=z, m=m, xhat=xhat, fft_cfg=loss_cfg.fft, weights=loss_cfg.weights, returnSub=returnSub)
        case signal_utils.SignalType.sine:
            return sine_loss(x=x, z=z, m=m, xhat=xhat, fft_cfg=loss_cfg.fft, weights=loss_cfg.weights, returnSub=returnSub)
        case default:
            raise ValueError("Invalid SignalType: {}".format(loss_cfg.type))

def _get_freq(x, xhat, m):
    """
        xhat ranges +-2m because xhat+z +-m 
        and z usally ranges +-0.5m but could be +-m when scaler = 1.99
        x usually ranges +-0.5m, but we keep consistent for recover_loss
        diving by 2m, although makes normalized xhat and x small, ensures they are [0, 1] for target_f_loss
    """
    freq_xhat = signal_utils.get_scaled_FFT(xhat, m*2+1)
    freq_x = signal_utils.get_scaled_FFT(x, m*2+1) # (batch_size, window_size//2+1)
    return freq_x, freq_xhat

def sine_loss(x, z, m, xhat, fft_cfg, weights, returnSub):
    '''
        x, xhat, z: (batch_size, window_size)
        m: (batch_size,)
        return: (batch_size,)
    '''
    window_size = fft_cfg.window_size
    bin_idx = int(fft_cfg.target_frequency/fft_cfg.bin_width)
    
    freq_x, freq_xhat = _get_freq(x=x, xhat=xhat, m=m)

    # 3 parts of loss
    target_f_loss = 1 - (freq_xhat[:, bin_idx])**2 #(batch_size,), [0, 1]
    
    non_target_freq_x = signal_utils.exclude_column(freq_x, bin_idx)
    non_target_freq_xhat = signal_utils.exclude_column(freq_xhat, bin_idx)
    recover_loss = torch.norm(non_target_freq_x - non_target_freq_xhat, dim = 1)/non_target_freq_x.shape[1] #(batch_size,), [0, 1]
    
    amplitude_loss = 1 - torch.norm((xhat+z)/m[:, None], dim = 1)/xhat.shape[1] #(batch_size,) [0, 1]; xhat+z and z is in +-m but xhat is not
    
    if log.getEffectiveLevel() is logging.DEBUG:
        utils.check_range([target_f_loss, recover_loss, amplitude_loss])
    
    tg, rc, am =  weights.target_f_loss*target_f_loss, weights.recover_loss*recover_loss, weights.amplitude_loss*amplitude_loss
    if returnSub:
        return tg + rc + am, (tg, rc, am)
    else:
        return tg + rc + am
    

def FMCW_loss(x, z, m, xhat, fft_cfg, weights, returnSub):
    window_size = fft_cfg.window_size
    bin_idx_l, bin_idx_r = [int(fft_cfg.target_frequency[key]/fft_cfg.bin_width) for key in ['start', 'end']]

    freq_x, freq_xhat = _get_freq(x=x, xhat=xhat, m=m)

    # 4 parts of loss
    target_f_loss = torch.norm(1-freq_xhat[:, bin_idx_l:bin_idx_r+1], dim=1)/(bin_idx_r-bin_idx_l+1) #(batch_size,), [0, 1]

    non_target_freq_x = signal_utils.exclude_column_range(freq_x, bin_idx_l, bin_idx_r)
    non_target_freq_xhat = signal_utils.exclude_column_range(freq_xhat, bin_idx_l, bin_idx_r)
    recover_loss = torch.norm(non_target_freq_x - non_target_freq_xhat, dim = 1)/non_target_freq_x.shape[1] #(batch_size,), [0, 1]
    
    amplitude_loss = 1 - torch.norm((xhat+z)/m[:, None], dim = 1)/xhat.shape[1] ##(batch_size,) [0, 1] ->#x+z, z is in +-m but x is not

    variance_loss = torch.std(freq_xhat[:, bin_idx_l:bin_idx_r+1], dim=1, correction=0)

    if log.getEffectiveLevel() is logging.DEBUG:
        utils.check_range([target_f_loss, recover_loss, amplitude_loss, variance_loss])
    
    tg, rc, am, vr = weights.target_f_loss*target_f_loss, weights.recover_loss*recover_loss, weights.amplitude_loss*amplitude_loss, weights.variance_loss*variance_loss

    if returnSub:
        return tg + rc + am + vr, (tg, rc, am, vr)
    else:
        return tg + rc + am + vr
