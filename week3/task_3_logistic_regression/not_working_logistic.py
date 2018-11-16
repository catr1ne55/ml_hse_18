import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score


def sigmoid(w, X):
  return 1./(1. + np.exp(-np.dot(X, w)))


def dist(x, y):
    return np.sqrt(np.square(x[0]-y[0]) + np.square(x[1]-y[1]))


def log_reg(x1, x2, y, k, w, C, e, steps):
    w1, w2 = w
    for i in range(steps):
        new_w1 = w1 + k * np.mean(y * x1 * (1. / (1. + np.exp(-y * (w1 * x1 + w2 * x2))))) - k * C * w1
        new_w2 = w2 + k * np.mean(y * x2 * (1. / (1. + np.exp(-y * (w1 * x1 + w2 * x2))))) - k * C * w2
        d = dist((new_w1, new_w2), (w1, w2))
        w1, w2 = new_w1, new_w2
        if d < e:
            break
    print(w1,'_', w2)
    preds = [1./(1. + np.exp(-w1*x1[i] - w2*x2[i])) for i in range(len(y))]
    return preds

data = pd.read_csv('data-logistic.csv', header=None)
y, x1, x2 = map(lambda x: data.loc[:, x], [0, 1, 2])

yt = log_reg(x1, x2, y, 0.1, [0, 0], 0, 0.00001, 10000)
yr = log_reg(x1, x2, y, 0.1, [0, 0], 10, 0.00001, 10000)

print(roc_auc_score(y, yt))
print(roc_auc_score(y, yr))