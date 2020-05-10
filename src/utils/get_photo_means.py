import numpy as np
import matplotlib.pyplot as plt


def get_photo_means(photo_path, art_path):
    photo = plt.imread(photo_path)
    photo = photo[:, :, :3]
    art = plt.imread(art_path)
    if len(photo.shape) == 2:
        photo = np.dstack((photo, photo, photo))
    if art.shape[2] == 4:
        art = art[:, :, 0:3]
    photo = photo[:, :, :3]
    means = np.mean(np.mean(photo, axis=1), axis=0).reshape((3, 1, 1))
    return means, photo, art
