import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import ResNet
from models.resnet_rnn import ResNetRNN
class FineResnetRnn(nn.Module):

    def __init__(self,pretrained_model:nn.Module,fixed_layer = 2):
        super(FineResnetRnn, self).__init__()
        self.fixed_layer = fixed_layer
        self.pretrained_model = pretrained_model
        self._fixed_layer()

    def _fixed_layer(self):
        for param in self.pretrained_model.cnn.conv1.parameters():
            param.requires_grad = False
        for idx, child in enumerate(self.pretrained_model.cnn.layers.children()):
            if idx < self.fixed_layer:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, images):
        # shape of images: (batch, channel, height, width)
        return self.pretrained_model(images)

    def name(self):
        return "fine_resnet_rnn"



if __name__ == '__main__':
    crnn = ResNetRNN(input_shape=(3,32,100),num_class= 63)
    #print(crnn.cnn.conv1)
    for param in crnn.cnn.conv1.parameters():
        param.requires_grad = False
    fixed_layer =  2
    for idx, child in enumerate(crnn.cnn.layers.children()):
        if idx < fixed_layer:
            for param in child.parameters():
                param.requires_grad = False
            print(child, idx)


