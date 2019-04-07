import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# MNIST_Parser
from mlxtend.data import loadlocal_mnist

import Knn as knn
import LDA as lda

# Read Mnist data
train_images, train_labels = loadlocal_mnist(
        images_path='data/train-images.idx3-ubyte',
        labels_path='data/train-labels.idx1-ubyte')

test_images, test_labels = loadlocal_mnist(
        images_path='data/t10k-images.idx3-ubyte',
        labels_path='data/t10k-labels.idx1-ubyte')


def show_numbers_image(n_rep, imgs, labels):
        matrix_images = np.empty((10, n_rep), dtype=object)
        for n in range(0, 10):
                for count, index_of_number in enumerate(np.where(labels == n)[0][:n_rep]):
                        matrix_images[n][count] = imgs[index_of_number].reshape(28, 28)

        imgs_show = np.vstack([np.hstack(row) for row in matrix_images])
        plt.imshow(imgs_show, cmap='gray', vmin=0, vmax=255)
        plt.show()


def show_number_image(img):
        plt.imshow(img.reshape(28, 28), cmap='gray', vmin=0, vmax=255)
        plt.show()
