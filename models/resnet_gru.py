import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import ResNet

class ResNetRRU(nn.Module):

    def __init__(self,input_shape,num_class,map_to_seq_hidden= 2048, rnn_hidden=128, leaky_relu=False):
        super(ResNetRRU, self).__init__()
        self.cnn = ResNet(input_shape)
        self.rnn = nn.GRU(map_to_seq_hidden, rnn_hidden, num_layers=2, bidirectional=True)
        self.dense = nn.Linear(rnn_hidden*2,num_class)

    def forward(self, images):
        # shape of images: (batch, channel, height, width)
        x = self.cnn(images)
        batch, channel, height, width = x.size()
        x = x.view(batch, channel * height, width)
        x = x.permute(2, 0, 1)  # (width, batch, feature)
        # seq = self.map_to_seq(conv)
        x, _ = self.rnn(x)
        output = self.dense(x)
        return output  # shape: (seq_len, batch, num_class)

    def name(self):
        return "resnet_gru"



if __name__ == '__main__':
    data = torch.rand((128, 3, 32, 100))
    crnn = ResNetRRU((3, 32, 100), 63)
    print(crnn)
    out = crnn(data)
    print(out.shape)