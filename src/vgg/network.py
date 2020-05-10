import pickle
import os

import lasagne
from lasagne.layers import InputLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def initialize_network(width):
    net = {}
    net["input"] = InputLayer((1, 3, width, width))
    net["conv1_1"] = ConvLayer(net["input"], 64, 3, pad=1)
    net["conv1_2"] = ConvLayer(net["conv1_1"], 64, 3, pad=1)
    net["pool1"] = PoolLayer(net["conv1_2"], 2, mode="average_exc_pad")
    net["conv2_1"] = ConvLayer(net["pool1"], 128, 3, pad=1)
    net["conv2_2"] = ConvLayer(net["conv2_1"], 128, 3, pad=1)
    net["pool2"] = PoolLayer(net["conv2_2"], 2, mode="average_exc_pad")
    net["conv3_1"] = ConvLayer(net["pool2"], 256, 3, pad=1)
    net["conv3_2"] = ConvLayer(net["conv3_1"], 256, 3, pad=1)
    net["conv3_3"] = ConvLayer(net["conv3_2"], 256, 3, pad=1)
    net["conv3_4"] = ConvLayer(net["conv3_3"], 256, 3, pad=1)
    net["pool3"] = PoolLayer(net["conv3_4"], 2, mode="average_exc_pad")
    net["conv4_1"] = ConvLayer(net["pool3"], 512, 3, pad=1)
    net["conv4_2"] = ConvLayer(net["conv4_1"], 512, 3, pad=1)
    net["conv4_3"] = ConvLayer(net["conv4_2"], 512, 3, pad=1)
    net["conv4_4"] = ConvLayer(net["conv4_3"], 512, 3, pad=1)
    net["pool4"] = PoolLayer(net["conv4_4"], 2, mode="average_exc_pad")
    net["conv5_1"] = ConvLayer(net["pool4"], 512, 3, pad=1)
    net["conv5_2"] = ConvLayer(net["conv5_1"], 512, 3, pad=1)
    net["conv5_3"] = ConvLayer(net["conv5_2"], 512, 3, pad=1)
    net["conv5_4"] = ConvLayer(net["conv5_3"], 512, 3, pad=1)
    net["pool5"] = PoolLayer(net["conv5_4"], 2, mode="average_exc_pad")

    return net


def build_model(width):
    net = initialize_network(width)
    with open(BASE_DIR + "/vgg/vgg19_normalized.pkl", "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        p = u.load()
        values = p["param values"]

    lasagne.layers.set_all_param_values(net["pool5"], values)
    return net
