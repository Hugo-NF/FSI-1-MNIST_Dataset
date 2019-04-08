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


# Accuracy per features combinations for 60-neighbors
# from itertools import combinations
#
# data = range(7)
# arr_combinations = []
# accuracy_per_combinations = []
#
# for i in range(1, len(data)+1):
#     arr_combinations = arr_combinations + list(combinations(data, i))
#
# for index, comb in enumerate(arr_combinations):
#     knn.set_n_neighbors(60)
#     knn.classifier_with_features(train_images, train_labels, list(comb))
#     accuracy = knn.get_accuracy(test_images, test_labels)
#     accuracy_per_combinations.append((accuracy, comb))
#     print({'combination': comb,
#            'accuracy': accuracy})
#
# print(max(accuracy_per_combinations))


# Test best features time to train and test, and accuracy:
import time


# Get statics for Knn, 3 neighbors and using best features for knn-3
print("KNN-3-best-features:")
knn.set_n_neighbors(3)

train_start_time = time.time()
knn.classifier_with_features(train_images, train_labels, [2, 3, 4, 5, 6])
train_finish_time = time.time()

test_start_time = time.time()
accuracy = knn.get_accuracy(test_images, test_labels)
test_finish_time = time.time()

print("Time Train-60000 (s): {}".format(train_finish_time - train_start_time))
print("Time Test-10000 (s):  {}".format(test_finish_time - test_start_time))
print("Accuracy: {}".format(accuracy))
knn.show_confusion_matrix(test_images, test_labels, "KNN-3-best-features-confusion-matrix.png")


# Get statics for Knn, 60 neighbors and using best features for knn-60
print("KNN-60-best-features:")
knn.set_n_neighbors(60)

train_start_time = time.time()
knn.classifier_with_features(train_images, train_labels, [2, 4, 5, 6])
train_finish_time = time.time()

test_start_time = time.time()
accuracy = knn.get_accuracy(test_images, test_labels)
test_finish_time = time.time()

print("Time Train-60000 (s): {}".format(train_finish_time - train_start_time))
print("Time Test-10000 (s):  {}".format(test_finish_time - test_start_time))
print("Accuracy: {}".format(accuracy))
knn.show_confusion_matrix(test_images, test_labels, "KNN-60-best-features-confusion-matrix.png")


# Get statics for LDA using image pixels as data
print("LDA-basic-features:")

train_start_time = time.time()
lda.basic_classifier(train_images, train_labels)
train_finish_time = time.time()

test_start_time = time.time()
accuracy = lda.get_accuracy(test_images, test_labels)
test_finish_time = time.time()

print("Time Train-60000 (s): {}".format(train_finish_time - train_start_time))
print("Time Test-10000 (s):  {}".format(test_finish_time - test_start_time))
print("Accuracy: {}".format(accuracy))
lda.show_confusion_matrix(test_images, test_labels, "LDA-basic-features-confusion-matrix.png")


# Get statics for LDA using best features
print("LDA-best-features:")

train_start_time = time.time()
lda.classifier_with_features(train_images, train_labels, [0, 1, 2, 3, 4, 5, 6])
train_finish_time = time.time()

test_start_time = time.time()
accuracy = lda.get_accuracy(test_images, test_labels)
test_finish_time = time.time()

print("Time Train-60000 (s): {}".format(train_finish_time - train_start_time))
print("Time Test-10000 (s):  {}".format(test_finish_time - test_start_time))
print("Accuracy: {}".format(accuracy))
lda.show_confusion_matrix(test_images, test_labels, "LDA-best-features-confusion-matrix.png")


# Get statics for Knn, 3 neighbors and using image pixels as data
print("KNN-3-basic-features:")
knn.set_n_neighbors(3)

train_start_time = time.time()
knn.basic_classifier(train_images, train_labels)
train_finish_time = time.time()

test_start_time = time.time()
accuracy = knn.get_accuracy(test_images, test_labels)
test_finish_time = time.time()

print("Time Train-60000 (s): {}".format(train_finish_time - train_start_time))
print("Time Test-10000 (s):  {}".format(test_finish_time - test_start_time))
print("Accuracy: {}".format(accuracy))
knn.show_confusion_matrix(test_images, test_labels, "KNN-3-basic-features-confusion-matrix.png")