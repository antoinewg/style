import lasagne
import numpy as np
import pickle
import skimage.transform
import scipy

import theano
import theano.tensor as T

from lasagne.utils import floatX

import matplotlib.pyplot as plt

from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer

from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.nonlinearities import softmax

from vgg.network import build_model
from vgg.functions import precompute_activations, tv_loss

from utils.get_photo_means import get_photo_means
from utils.prepare_image import prepare_image
from utils.rollback import roll_back

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(BASE_DIR, "src/images")

IMAGE_W = 120

net = build_model(IMAGE_W)

photo_path = os.path.join(IMAGES_DIR, "photo.jpg")
art_path = os.path.join(IMAGES_DIR, "mona_lisa.jpg")
means, photo, art = get_photo_means(photo_path, art_path)

photo = prepare_image(photo, IMAGE_W, means)
art = prepare_image(art, IMAGE_W, means)

# precompute layer activations
layers = ["conv4_2", "conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]
layers = {k: net[k] for k in layers}
photo_features, art_features = precompute_activations(layers, photo, art)

# Get expressions for layer activations for generated image
lim = 128
generated = theano.shared(
    floatX(np.random.uniform(-1 * lim, lim, (1, 3, IMAGE_W, IMAGE_W)))
)

gen_features = lasagne.layers.get_output(layers.values(), generated)
gen_features = {k: v for k, v in zip(layers.keys(), gen_features)}



def content_loss(X, Y, layer):
    return 1.0 / 2 * ((Y[layer] - X[layer]) ** 2).sum()


def gram_mat(vecs):
    vecs = vecs.flatten(ndim=3)
    gram = T.tensordot(vecs, vecs, axes=([2], [2]))
    return gram


def style_loss(X, Y, layer):
    x = X[layer]
    y = Y[layer]

    N = x.shape[1]
    M = x.shape[2] * x.shape[3]

    loss = 1.0 / (4 * N ** 2 * M ** 2) * ((gram_mat(y) - gram_mat(x)) ** 2).sum()
    return loss


def define_global_loss(
    photo_features, gen_features, art_features, cl_scalar, sl_scalar, tv_scalar, tv_pow
):
    # define total loss

    losses = []
    losses.append(cl_scalar * content_loss(photo_features, gen_features, "conv4_2"))
    losses.append(sl_scalar * style_loss(art_features, gen_features, "conv1_1"))
    losses.append(sl_scalar * style_loss(art_features, gen_features, "conv2_1"))
    losses.append(sl_scalar * style_loss(art_features, gen_features, "conv3_1"))
    losses.append(sl_scalar * style_loss(art_features, gen_features, "conv4_1"))
    losses.append(sl_scalar * style_loss(art_features, gen_features, "conv5_1"))
    losses.append(tv_scalar * tv_loss(generated, tv_pow))
    total_loss = sum(losses)
    grad = T.grad(total_loss, generated)
    f_loss = theano.function([], total_loss)
    f_grad = theano.function([], grad)

    return f_loss, f_grad


# define loss functions
f_loss, f_grad = define_global_loss(
    photo_features,
    gen_features,
    art_features,
    cl_scalar=0.001,
    sl_scalar=2e5,
    tv_scalar=1e-8,
    tv_pow=1.25,
)

# start from random noise
generated.set_value(floatX(np.random.uniform(-1 * lim, lim, (1, 3, IMAGE_W, IMAGE_W))))
x0 = generated.get_value().astype("float64")
xs = []
xs.append(x0)


def eval_loss(x0, width):
    x0 = floatX(x0.reshape((1, 3, width, width)))
    generated.set_value(x0)
    return f_loss().astype("float64")


def eval_grad(x0, width):
    x0 = floatX(x0.reshape((1, 3, width, width)))
    generated.set_value(x0)
    return np.array(f_grad()).flatten().astype("float64")


# optimize, saving the result periodically
iters = 2
for i in range(iters):
    print(f"Iteration {i}")
    scipy.optimize.fmin_l_bfgs_b(
        eval_loss, x0.flatten(), fprime=eval_grad, args=(IMAGE_W,), maxfun=40
    )
    x0 = generated.get_value().astype("float64")
    xs.append(x0)


plt.figure(figsize=(12, 12))
for i in range(iters):
    plt.subplot(3, 4, i + 1)
    plt.gca().xaxis.set_visible(False)
    plt.gca().yaxis.set_visible(False)
    plt.imshow(roll_back(xs[i], means))
plt.tight_layout()
plt.savefig("progress.png")

plt.figure(figsize=(8, 8))
plt.imshow(roll_back(xs[-1], means), interpolation="nearest")
plt.savefig("neural_painting.png")
