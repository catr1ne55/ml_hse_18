import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score


data = pd.read_csv('data-logistic.csv', header=None)
X = data.values[:, 1:]
y = data.values[:, :1].T[0]


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def dist(a, b):
    return np.sqrt(np.square(a[0] - b[0]) + np.square(a[1] - b[1]))


def logreg(X, y, k, w, C, e, steps):
    w1, w2 = w
    for i in range(steps):
        w1n = w1 + k * np.mean(y * X[:, 0] * (1 - (1. / (1 + np.exp(-y * (w1 * X[:, 0] + w2 * X[:, 1])))))) - k * C * w1
        w2n = w2 + k * np.mean(y * X[:, 1] * (1 - (1. / (1 + np.exp(-y * (w1 * X[:, 0] + w2 * X[:, 1])))))) - k * C * w2
        if dist((w1n, w2n), (w1, w2)) < e:
            break
        w1 = w1n
        w2 = w2n

    preds = []
    for i in range(len(X)):
        t1 = w1 * X[i, 0] + w2 * X[i, 1]
        s = sigmoid(t1)
        preds.append(s)
    return preds


p0 = logreg(X, y, 0.1, [0.0, 0.0], 0, 0.00001, 10000)
p1 = logreg(X, y, 0.1, [0.0, 0.0], 10, 0.00001, 10000)

print(roc_auc_score(y, p0))
print(roc_auc_score(y, p1))