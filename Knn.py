import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

import features as ft

neigh = KNeighborsClassifier(n_neighbors=3)
train_X = None
train_y = None
last_classifier_used = None
last_features_used = None


def set_n_neighbors(n):
    global neigh
    neigh = KNeighborsClassifier(n_neighbors=n)


def basic_classifier(t_X, t_y):
    global train_X, train_y, last_classifier_used
    last_classifier_used = basic_classifier
    train_X = t_X
    train_y = t_y
    neigh.fit(train_X, train_y)


# index_features_function_value:
#   0 - centroid
#   1 - axis_least_inertia
#   2 - eccentricity
#   3 - circularity_ratio
#   4 - rectangularity
#   5 - convexity
#   6 - solidity
def classifier_with_features(t_X, t_y, index_features_functions=None):
    global train_X, train_y, last_features_used, last_classifier_used

    last_classifier_used = classifier_with_features

    if index_features_functions != None:
        last_features_used = index_features_functions

    train_X = t_X
    train_y = t_y
    neigh.fit(get_new_X(train_X), train_y)


def get_predict(X):
    if last_classifier_used == classifier_with_features:
        return neigh.predict(get_new_X(X))
    else:
        return neigh.predict(X)


def get_predict_proba(X):
    if last_classifier_used == classifier_with_features:
        return neigh.predict_proba(get_new_X(X))
    else:
        return neigh.predict_proba(X)


def show_accuracy(test_X, test_y):
    predicts = get_predict(test_X)
    print(accuracy_score(test_y, predicts))


def show_confusion_matrix(test_X, test_y, name_file='knn_confusion_matrix.png'):
    labels = np.unique(test_y)
    predicts = get_predict(test_X)
    plot_confusion_matrix(test_y, predicts, labels, title=name_file.replace('.png', ''))
    plt.savefig(name_file, dpi=300)
    plt.show()


def show_classification_report(test_X, test_y):
    labels = np.unique(test_y)
    predicts = get_predict(test_X)
    print(classification_report(test_y, predicts, labels))


def plot_accuracy_per_neighbors(test_X, test_y, neighbor_range, plot_confusion=True, limit=2000):
    train_error = np.zeros(len(neighbor_range))
    test_error = np.zeros(len(neighbor_range))
    labels = np.unique(test_y[:limit])

    for index, neighbor in enumerate(neighbor_range):
        set_n_neighbors(neighbor)
        last_classifier_used(train_X, train_y)

        predicts = get_predict(train_X[:limit])
        train_error[index] = 1 - accuracy_score(train_y[:limit], predicts)

        predicts = get_predict(test_X[:limit])
        test_error[index] = 1 - accuracy_score(test_y[:limit], predicts)

        # Plot confusion matrix
        if plot_confusion:
            plot_confusion_matrix(test_y[:limit], predicts, labels,
                                  title='Confusion Matriz - {} Neighbors'.format(neighbor))

            plt.savefig('knn_confusion_matrix_{}.png'.format(neighbor), dpi=300)
            plt.show()

    # plot train_error and test_error in the graph
    df = pd.DataFrame({'x': neighbor_range, 'train-error': train_error, 'test-error': test_error})
    plt.plot('x', 'train-error', data=df, marker='', color='blue', linewidth=2)
    plt.plot('x', 'test-error', data=df, marker='', color='red', linewidth=2)
    plt.xlabel("Nº of Neighbors")
    plt.ylabel("Predict Error")
    plt.legend()
    plt.savefig('accuracy_per_neighbors.png', dpi=300)
    plt.show()


def get_new_X(old_X):
    new_t_X = [[] for i in range(len(old_X))]

    for index, X in enumerate(old_X):

        feature = ft.Metrics(X)
        arr_features_functions = [feature.centroid,
                                  feature.axis_least_inertia,
                                  feature.eccentricity,
                                  feature.circularity_ratio,
                                  feature.rectangularity,
                                  feature.convexity,
                                  feature.solidity]

        for index_feature_function in last_features_used:
            value = arr_features_functions[index_feature_function]()
            if type(value) == int or type(value) == float:
                value = [value]
            elif type(value) == np.ndarray:
                value = value.tolist()

            new_t_X[index] = new_t_X[index] + value

    return new_t_X

# Function copied from: scikit-learn: plot_confusion_matrix
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
