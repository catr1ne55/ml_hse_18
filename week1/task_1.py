import numpy as np

X = np.random.normal(loc=1, scale=10, size=(1000, 50))
X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

Z = np.array([[4, 5, 0],
             [1, 9, 3],
             [5, 1, 1],
             [3, 3, 3],
             [9, 9, 9],
             [4, 7, 1]])

Z_sum = np.sum(Z, axis=1)
Z_bool = np.nonzero(Z_sum > 10)
print(Z_bool)

e1 = np.eye(3)
e2 = np.eye(3)
e = np.vstack((e1, e2))
print(e)