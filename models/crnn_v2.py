import torch
import torch.nn as nn
from collections import OrderedDict

class CRNN_V2(nn.Module):
    """
    Simple CRNN
    """
    def __init__(self,input_shape, num_class,map_to_seq_hidden= 256, rnn_hidden=128, leaky_relu=False):
        super(CRNN_V2, self).__init__()
        #self.input_shape = (img_channel,img_height,img_width)
        self.input_shape = input_shape
        channels = [32, 64, 128, 256, 256]
        layers = [2, 2, 2, 2, 2]
        kernels = [3, 3, 3, 3, 3]
        pools = [2, 2, 2, 2, (2, 1)]
        modules = OrderedDict()
        def _cba(name, in_channels, out_channels, kernel_size):
            modules[f'conv{name}'] = nn.Conv2d(in_channels, out_channels, kernel_size,
                                               padding=(1, 1) if kernel_size == 3 else 0)
            modules[f'bn{name}'] = nn.BatchNorm2d(out_channels)
            modules[f'relu{name}'] = nn.ReLU(inplace=True)

        last_channel = 3
        for block, (n_channel, n_layer, n_kernel, k_pool) in enumerate(zip(channels, layers, kernels, pools)):
            for layer in range(1, n_layer + 1):
                _cba(f'{block + 1}{layer}', last_channel, n_channel, n_kernel)
                last_channel = n_channel
            modules[f'pool{block + 1}'] = nn.MaxPool2d(k_pool)
        modules[f'dropout'] = nn.Dropout(0.25, inplace=True)

        self.cnn = nn.Sequential(modules)
        self.lstm = nn.LSTM(input_size= map_to_seq_hidden, hidden_size=rnn_hidden, num_layers=2, bidirectional=True)
        self.dense = nn.Linear(in_features=2*rnn_hidden, out_features=num_class)


    def forward(self, x):
        # shape of images: (batch, channel, height, width)
        x = self.cnn(x)
        batch, channel, height, width = x.size()
        x = x.view(batch, channel * height, width)
        x = x.permute(2, 0, 1)  # (width, batch, feature)
        x, _ = self.lstm(x)
        x = self.dense(x)
        # shape: (seq_len, batch, num_class)
        return x

    def name(self):
        return "crnn_v2"

if __name__ == '__main__':
    data = torch.rand((128, 3, 32, 100))
    crnn = CRNN_V2((3, 32, 100), 63)
    print(crnn)
    out = crnn(data)
    print(data.shape)