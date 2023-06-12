
from .nnet import NNet

def nnet():
    net = NNet(1, 32, 5, 2)
    net.use_2dconv = False
    net.bandwise = False
    return net


