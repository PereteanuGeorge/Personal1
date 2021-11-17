import torch
from torch import nn


def square(input):
    '''
    Applies the Sigmoid Linear Unit (SiLU) function element-wise:
        SiLU(x) = x * sigmoid(x)
    '''
    return torch.mul(input, input)


# create a class wrapper from PyTorch nn.Module, so
# the function now can be easily used in models
class SquareActivation(nn.Module):
    '''
    Applies the Sigmoid Linear Unit (SiLU) function element-wise:
        SiLU(x) = x * sigmoid(x)
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    References:
        -  Related paper:
        https://arxiv.org/pdf/1606.08415.pdf
    Examples:
    '''

    def __init__(self):
        '''
        Init method.
        '''
        super().__init__()  # init the base class

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return square(input)  # simply apply already implemented SiLU
