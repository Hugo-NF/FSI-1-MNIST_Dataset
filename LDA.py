import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import features as ft

clf = LinearDiscriminantAnalysis()
train_X = None
train_y = None
last_classifier_used = None
last_features_used = None


def basic_classifier(t_X, t_y):
    global train_X, train_y, last_classifier_used
    last_classifier_used = basic_classifier
    train_X = t_X
    train_y = t_y
    clf.fit(train_X, train_y)


# index_features_function_value:
#   0 - centroid
#   1 - axis_least_inertia
#   2 - eccentricity
#   3 - circularity_ratio
#   4 - rectangularity
#   5 - convexity
#   6 - solidity
def classifier_with_features(t_X, t_y, index_features_functions=None):
    global last_features_used, last_classifier_used

    last_classifier_used = classifier_with_features

    new_t_X = [[] for i in range(len(t_X))]

    if index_features_functions != None:
        last_features_used = index_features_functions

    for index, X in enumerate(t_X):

        feature = ft.Metrics(X)
        arr_features_functions = [feature.centroid,
                                  feature.axis_least_inertia,
                                  feature.eccentricity,
                                  feature.circularity_ratio,
                                  feature.rectangularity,
                                  feature.convexity,
                                  feature.solidity]

        for index_feature_function in index_features_functions:
            value = arr_features_functions[index_feature_function](X)
            if type(value) == int or type(value) == float:
                value = [value]
            elif type(value) == np.ndarray:
                value = value.tolist()

            new_t_X[index] = new_t_X[index] + value

    train_X = new_t_X
    train_y = t_y
    clf.fit(train_X, train_y)


def show_predict(X):
    predicts = clf.predict(X)
    print(predicts)


def show_predict_proba(X):
    print(clf.predict_proba(X))


def show_accuracy(test_X, test_y):
    predicts = clf.predict(test_X)
    print(accuracy_score(test_y, predicts))


def show_confusion_matrix(test_X, test_y, name_file='lda_confusion_matrix.png'):
    labels = np.unique(test_y)
    predicts = clf.predict(test_X)
    plot_confusion_matrix(test_y, predicts, labels, title=name_file.replace('.png', ''))
    plt.savefig(name_file, dpi=300)
    plt.show()


def show_classification_report(test_X, test_y):
    labels = np.unique(test_y)
    predicts = clf.predict(test_X)
    print(classification_report(test_y, predicts, labels))
    
    
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