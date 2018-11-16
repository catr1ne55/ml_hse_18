import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import math

data = pd.read_csv('data-logistic.csv', header=None)
y = data.loc[:, 0]
X = data.loc[:, 1:]

c = 10
k = 0.1
w = [[0.0, 0.0]]


def delta(X, y, w, c, k, m):
    dp = np.einsum('ij,j->i', X, w[m - 1])
    coeff = 1 - (1. / (1 + np.exp(-y * dp)))
    a = k / coeff.shape[0] * (np.einsum('ij,i,i->j', X, y, coeff))
    w0 = w[m - 1][0] + a[0] - k * c * w[m - 1][0]
    w1 = w[m - 1][1] + a[1] - k * c * w[m - 1][1]
# w0 = w[m-1][0] + a[0] - без регуляризации
# w1 = w[m-1][1] + a[1] - без регуляризации
    w.append([w0, w1])
    return w

e = 100000
cnt = 0
i = 1
while e > 0.00001 and cnt < 100000:
    w = delta(X, y, w, c, k, i)
    e = math.sqrt((w[i][0] - w[i - 1][0]) ** 2 + (w[i][1] - w[i - 1][1]) ** 2)
    i += 1
    cnt += 1
    # w = [0.02856196551701265, 0.024783655436404754] w = [0.28781162047177644, 0.091983302159254335]
    # dp = np.einsum('ij,j->i',X,w)
#y_w = X * w
# coeff = 1/(1+np.exp(-y_w[:,0]-y_w[:,1]))
coeff = 1. / (1 + np.exp(-y * np.dot(X, w)))
y_true = np.array(y)
y_scores = np.array(coeff)

res = roc_auc_score(y_true, y_scores)
print(res)
print(np.around(res, 3))
