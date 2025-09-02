import torch
import torch.nn as nn
import torchaudio
from torchaudio.prototype.functional import sinc_impulse_response
from utils import utils

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn_layers = nn.Sequential(
            #input: (n, 1，2, 512)
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding='valid'), #output: (n, 1, 1, 511)
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)), #(n, 1, 1, 256)
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(255, 128),
            nn.Linear(128, 512),
        )
        self.activation_layers = nn.Sequential(
            nn.Tanh()
        )
        self.initSincFilter()

    def myClip(self, x):
#         x = x.type(torch.int32).type(torch.float64) #x+z is 16
#         x = torch.clip(x, min = torch.min(), max = m), [max(-m, -m-z), min(m-z, m)]
        rounded_x = torch.round(x)
        x = (rounded_x-x).detach() + x #(n, 512)
        return x

    def myNorm(self, x, m, z):
        return x * m[:, None] - z #(n, 512)

    def initSincFilter(self):
        cutoff, fs, ws = [18e3, 20e3], 48e3, 512
        cutoff = torch.tensor(cutoff)/fs*2
        ws -= 1 # the window size has to be odd
        self.irs = sinc_impulse_response(cutoff[0], window_size=ws) - sinc_impulse_response(cutoff[1], window_size=ws) #(1, ws), [-2, 2], band-pass
        if torch.cuda.is_available():
            self.irs = self.irs.cuda()

    def block(self, x, z, m):
        x = torch.unsqueeze(x, dim = 1) #output: (n, 1, 512)
        x = torch.cat((x, torch.unsqueeze(z, dim = 1)), dim = 1) #(n, 2, 512)
        x = torch.unsqueeze(x, dim = 1) #(n, 1, 2, 512)
        x = self.cnn_layers(x) #(n, 1, 1, 255)
        x = x.view(x.size(0), -1) #(n, 255)
        x = self.linear_layers(x) #(n, 512)

        x = torchaudio.functional.convolve(x, self.irs, mode='same') #/512
        
        x = self.activation_layers(x) #(n, 512), [0,1]
        x = self.myNorm(x, m , z)
        x = self.myClip(x) #(n, 512)
        
        #utils.check_range(x+z, -m[0], m[0])

        return x
        
    def forward(self, x, z, m):
        '''
            x: (n, 512) # n is batch size
            z: (n, 512)
            m: (n,)
            return: (n, 512)
        '''
        x = self.block(x, z, m)
        
        return x
