import torch
from torch import nn

from utils import device, load_weights
import tenseal as ts

class ConvNet2(torch.nn.Module):
    def __init__(self):
        super(ConvNet2, self).__init__()
        self.linear9 = nn.Linear(128, 64)
        self.linear10 = nn.Linear(64, 10)

    def forward(self, x):
        out = self.linear9(x)
        out = self.linear10(out)
        return out

model2 = ConvNet2().to(device)

load_weights(model2)

model2.eval()

class EncConvNet:
    def __init__(self, torch_nn):
        super(EncConvNet, self).__init__()

        self.fc9_weight = torch_nn.linear9.weight.T.data.tolist()
        self.fc9_bias = torch_nn.linear9.bias.data.tolist()

        self.fc10_weight = torch_nn.linear10.weight.T.data.tolist()
        self.fc10_bias = torch_nn.linear10.bias.data.tolist()

    def forward(self, enc_x):
        enc_x = ts.CKKSVector.pack_vectors(enc_x)
        enc_x = enc_x.mm(self.fc9_weight) + self.fc9_bias
        enc_x = enc_x.mm(self.fc10_weight) + self.fc10_bias
        return enc_x

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

enc_model = EncConvNet(model2)

from square_activation import square

class ConvNet3(torch.nn.Module):
    def __init__(self):
        super(ConvNet3, self).__init__()
        self.base = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=True)

        self.layer1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=True)

        self.layer2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        out = self.base(x)
        out = out * out
        out = self.layer1(out)
        out = out * out
        out = self.layer2(out)
        out = out * out
        return out

model3 = ConvNet3().to(device)

load_weights(model3)


class EncConvNet3:
    def __init__(self, torch_nn):
        super(EncConvNet3, self).__init__()

        self.conv1_weight = torch_nn.base.weight.data.view(
            torch_nn.base.out_channels, -1
        ).tolist()
        self.conv1_bias = torch_nn.base.bias.data.tolist()

        self.conv2_weight = torch_nn.layer1.weight.data.view(
            torch_nn.layer1.out_channels, -1
        ).tolist()
        self.conv2_bias = torch_nn.layer1.bias.data.tolist()

        self.conv3_weight = torch_nn.layer2.weight.data.view(
            torch_nn.layer2.out_channels, -1
        ).tolist()
        self.conv3_bias = torch_nn.layer2.bias.data.tolist()

    def forward(self, enc_x, windows_nb):
        enc_channels = []
        for kernel, bias in zip(self.conv1_weight, self.conv1_bias):
            y = enc_x.conv2d_im2col(kernel, windows_nb) + bias
            enc_channels.append(y)
        enc_x = ts.CKKSVector.pack_vectors(enc_channels)
        enc_x.square_()
        for kernel, bias in zip(self.conv2_weight, self.conv2_bias):
            y = enc_x.conv2d_im2col(kernel, windows_nb) + bias
            enc_channels.append(y)
        enc_x = ts.CKKSVector.pack_vectors(enc_channels)
        enc_x.square_()
        for kernel, bias in zip(self.conv3_weight, self.conv3_bias):
            y = enc_x.conv2d_im2col(kernel, windows_nb) + bias
            enc_channels.append(y)
        enc_x = ts.CKKSVector.pack_vectors(enc_channels)
        enc_x.square_()
        return enc_x

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

enc_model3 = EncConvNet3(model3)