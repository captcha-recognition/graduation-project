import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):

    def __init__(self, in_channel, out_channel, stride=1):
        super().__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel, track_running_stats=True)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel, track_running_stats=True)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.in_channel = 64
        channels = [64,128,256,512]
        strides = [1,2,2,2]
        num_blocks = [2,2,2,2]
        self.input_shape = input_shape
        channel, height, width = input_shape
        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, self.in_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channel, track_running_stats=True),
            nn.ReLU(),
        )
        layers = [self.make_layer(ResidualBlock,channel,num_block,stride) for channel,stride,num_block in zip(channels,strides,num_blocks)]
        self.layers = nn.Sequential(*layers)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, channels, stride))
            self.in_channel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)  # [3, 32, 32] -> [64, 32, 32]
        x = self.layers(x) #  [64, 32, 32] -> [512, 4, 4]
        x = F.avg_pool2d(x, 4)  # -> [512, 1, 1]
        x = x.view(-1, 512)  # -> [1, 512]
        return x


if __name__ == '__main__':
    resnet = ResNet((3,32,100))
    #(batch, channel, height, width)
    x = torch.rand((64,3,32,100))
    out = resnet(x)