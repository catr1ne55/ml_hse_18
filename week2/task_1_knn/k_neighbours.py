from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale
import pandas as pd


y = pd.read_csv('wine.data', usecols=[0], header=None)
X = pd.read_csv('wine.data', usecols=[1,2,3,4,5,6,7,8,9,10,11,12,13], header=None)

X1 = scale(X)

gen = KFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = []
cv_scores1 = {}

for k in range(1, 50):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=gen, scoring='accuracy')
    scores1 = cross_val_score(knn, X1, y, cv=gen, scoring='accuracy')
    cv_scores.append(scores.mean())
    cv_scores1[k] = scores1.mean()

max_score = max(cv_scores)
max_score1 = max(cv_scores1.values())
print(cv_scores.index(max_score), max_score)
print(max_score1)
print(cv_scores1)


