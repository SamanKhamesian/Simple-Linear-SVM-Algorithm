import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score

from Source.svm import SVM

N_SAMPLE = 200
SPLIT_RATIO = 0.6
N_FEATURES = 2


def plot_dataset(x, y, w):
    plt.xlabel('Feature 01')
    plt.ylabel('Feature 02')
    plt.title('Our Dataset')

    plt.scatter(x=x[y == -1, 1],
                y=x[y == -1, 2],
                edgecolors='black',
                label='Class 01')

    plt.scatter(x=x[y == 1, 1],
                y=x[y == 1, 2],
                edgecolors='black',
                label='Class 02')

    min_x = np.min(x[:, 1])
    max_x = np.max(x[:, 1])

    A = w[0][0][0]
    B = w[1][0][0]
    C = w[2][0][0]

    point_1 = [min_x, max_x]
    point_2 = [-(A + B * min_x) / C, -(A + B * max_x) / C]

    plt.legend()
    plt.plot(point_1, point_2, 'k')
    plt.show()


def make_dataset():
    x, y = make_blobs(n_samples=N_SAMPLE,
                      n_features=N_FEATURES,
                      centers=2,
                      random_state=False,
                      shuffle=True,
                      cluster_std=0.6)

    y[y == 0] = -1
    bias = np.ones((1, N_SAMPLE))
    x = np.insert(x, 0, bias, axis=1)

    cut = int(N_SAMPLE * SPLIT_RATIO)
    x_train = x[:cut, :]
    x_test = x[cut:, :]
    y_train = y[:cut]
    y_test = y[cut:]

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = make_dataset()
    model = SVM()
    model.fit(X_train, Y_train)
    weights = model.get_weights()
    predictions = model.predict(X_test)
    plot_dataset(X_train, Y_train, weights)
    print('Accuracy is : %' + str(accuracy_score(Y_test, predictions) * 100))
