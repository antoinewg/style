import lasagne
from lasagne.utils import floatX
import theano
import theano.tensor as T


def precompute_activations(layers, base, style):
    input_im_theano = T.tensor4()
    outputs = lasagne.layers.get_output(layers.values(), input_im_theano)

    photo_feat = {
        k: theano.shared(output.eval({input_im_theano: base}))
        for k, output in zip(layers.keys(), outputs)
    }
    art_feat = {
        k: theano.shared(output.eval({input_im_theano: style}))
        for k, output in zip(layers.keys(), outputs)
    }

    return photo_feat, art_feat


def tv_loss(x, k):
    rev = x[:, :, :-1, :-1]
    return (((rev - x[:, :, 1:, :-1]) ** 2 + (rev - x[:, :, :-1, 1:]) ** 2) ** k).sum()
