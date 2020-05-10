import numpy as np
import skimage.transform

from lasagne.utils import floatX


def prepare_image(img, width, means):
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.repeat(img, 3, axis=2)

    height_img, width_img, _ = img.shape
    if height_img < width_img:
        new_height = int(width_img * width / height_img)
        img = skimage.transform.resize(img, (width, new_height), preserve_range=True)
    else:
        new_width = int(height_img * width / width_img)
        img = skimage.transform.resize(img, (new_width, width), preserve_range=True)

    # crop the center
    height_img, width_img, _ = img.shape
    img = img[
        height_img // 2 - width // 2 : height_img // 2 + width // 2,
        width_img // 2 - width // 2 : width_img // 2 + width // 2,
    ]

    # shuffle axes to c01
    img = np.swapaxes(np.swapaxes(img, 1, 2), 0, 1)

    # convert RGB to BGR
    img = img[::-1, :, :]

    # zero mean scaling
    img = img - means

    return floatX(img[np.newaxis])
