import sklearn.datasets
import sklearn.preprocessing
import sklearn.model_selection
from sklearn.neighbors import KNeighborsRegressor

import numpy as np

boston = sklearn.datasets.load_boston()
y = boston.target
X = sklearn.preprocessing.scale(boston.data)

gen = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=42)

scores_dict = {}

for p in np.linspace(1,10,200):
    knn = KNeighborsRegressor(n_neighbors=5, weights='distance', p=p, metric='minkowski')
    scores = sklearn.model_selection.cross_val_score(knn, X, y, cv=gen, scoring='neg_mean_squared_error')
    scores_dict[p] = scores.mean()

m_value = max(scores_dict.values())
print(scores_dict)
print(m_value)
print([key for key in scores_dict.keys() if scores_dict[key] == m_value])