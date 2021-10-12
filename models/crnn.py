import torch
import torch.nn as nn
from models.simple_cnn import SimpleCNN

class CRNN(nn.Module):

    def __init__(self,input_shape, num_class,
                 map_to_seq_hidden=64, rnn_hidden=256, leaky_relu=False):
        super(CRNN, self).__init__()
        (img_channel, img_height, img_width) = input_shape
        self.cnn = SimpleCNN(img_channel, img_height, img_width, leaky_relu)
        output_channel, output_height, output_width = self.cnn.features()
        self.map_to_seq = nn.Linear(output_channel * output_height, map_to_seq_hidden)
        self.rnn = nn.LSTM(map_to_seq_hidden, rnn_hidden, num_layers=2, bidirectional=True)
        self.dense = nn.Linear(2 * rnn_hidden, num_class)

    def forward(self, images):
        # shape of images: (batch, channel, height, width)
        conv = self.cnn(images)
        batch, channel, height, width = conv.size()
        conv = conv.view(batch, channel * height, width)
        conv = conv.permute(2, 0, 1)  # (width, batch, feature)
        seq = self.map_to_seq(conv)
        recurrent, _ = self.rnn(seq)
        output = self.dense(recurrent)
        return output  # shape: (seq_len, batch, num_class)

    def name(self):
        return "crnn"


if __name__ == '__main__':
    data = torch.rand((64,3,32,100))
    crnn = CRNN((3,32,100),63)
    print(crnn)
    out = crnn(data)
    print(out.shape)