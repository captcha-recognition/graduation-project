# cnn + rnn + ctc model
import torch
import torch.nn as nn
import torch.nn.functional as F
import bi_lstm

rnns = {
    'bi_lstm': bi_lstm,
    'lstm': nn.LSTM,
    'gru':nn.GRU
}

def make_rnn(name):
    return rnns[name]

class cnn_rnn_ctc(nn.Module):
    def __init__(self):
        super(cnn_rnn_ctc, self).__init__()
        self.cnn = nn.Sequential()
        self.rnn = nn.LSTM()

    def get_name(self):
        return "cnn_rnn_ctc"

    def forward(self,X):
        pass

    def cal_loss(self):
        pass

    def backward(self):
        pass