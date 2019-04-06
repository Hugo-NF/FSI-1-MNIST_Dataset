# Python version
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy as np
print('numpy: {}'.format(np.__version__))
# matplotlib
import matplotlib
import matplotlib.pyplot as plt
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))
# MNIST_Parser
from mlxtend.data import loadlocal_mnist

# Read Mnist data
train_images, trains_labels = loadlocal_mnist(
        images_path='data/train-images.idx3-ubyte',
        labels_path='data/train-labels.idx1-ubyte')


def show_numbers_image(n_rep):
        matrix_images = np.empty((10, n_rep), dtype=object)
        for n in range(0, 10):
                for count, index_of_number in enumerate(np.where(trains_labels == n)[0][:n_rep]):
                        matrix_images[n][count] = train_images[index_of_number].reshape(28, 28)

        return np.vstack([np.hstack(row) for row in matrix_images])


plt.imshow(show_numbers_image(10), cmap='gray', vmin=0, vmax=255)
plt.show()