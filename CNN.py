import torch
from torch import nn

IMAGE_SIZE = 28
N_CHANNELS = 1
N_CLASSES = 10
N_FEATURES = 16

class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=0, bias=True):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        x= self.conv(x)
        x= self.bn(x)
        return self.relu(x)


class ConvNet(nn.Module):
    def __init__(self, N_GPU, N_CLASSES):
        super(ConvNet, self).__init__()

        self.ngpu = N_GPU
        self.conv1 = BasicConv(N_CHANNELS, N_FEATURES, stride=2)
        self.conv2 = BasicConv(N_FEATURES, N_FEATURES * 2, stride=1)
        self.conv3 = BasicConv(N_FEATURES * 2, N_FEATURES * 4, stride=2)
        self.conv4 = BasicConv(N_FEATURES * 4, N_FEATURES * 8, stride=1)
        self.conv5 = BasicConv(N_FEATURES * 8, N_CLASSES, stride=1)

        self.sm = nn.Softmax()

    def forward(self, x):
        return self.sm(self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(x))))))