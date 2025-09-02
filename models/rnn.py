import torch
import torch.nn as nn
import torchaudio
from torchaudio.prototype.functional import sinc_impulse_response
from utils import utils


class RNN(nn.Module):
    def __init__(self, input_size=512, hidden_size=64, output_size=512, num_layers=2):
        super(RNN, self).__init__()

        # Encoder
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)

        # Decoder
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)

        # Output layer
        self.output_layer = nn.Linear(hidden_size, output_size)

        self.last_cnn = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=2, padding='same')

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
    def encoderDecoder(self, x):
        # Encoder
        encoder_output, (encoder_hidden, encoder_cell) = self.encoder(x)
        # Decoder
        decoder_output, _ = self.decoder(encoder_output, (encoder_hidden, encoder_cell))
        # Output layer
        x = self.output_layer(decoder_output) #(n, 2, 512)
        x = self.last_cnn(x)  #(n, 1, 512)
        
        return x

    def block(self, x, z, m):
        x = torch.unsqueeze(x, dim = 1) #output: (n, 1, 512)
        x = torch.cat((x, torch.unsqueeze(z, dim = 1)), dim = 1) #(n, 2, 512)

        x = self.encoderDecoder(x) #(n, 1, 512)
        
        x = x.view(x.size(0), -1) #(n, 512)

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
