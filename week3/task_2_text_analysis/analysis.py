from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVC
import numpy as np

newsgroups = datasets.fetch_20newsgroups(
                    subset='all',
                    categories=['alt.atheism', 'sci.space']
             )

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(newsgroups.data)
y = newsgroups.target

# grid = {'C': np.power(10.0, np.arange(-5, 6))}
# cv = KFold(n_splits=5, shuffle=True, random_state=241)
# clf = SVC(kernel='linear', random_state=241)
# gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
# gs.fit(X, y)
#
#
# max_score = gs.grid_scores_[0]
# max_c = max_score.mean_validation_score
# for a in gs.grid_scores_:
#     current = a.mean_validation_score
#     if current > max_c:
#         max_c = current
#         max_score = a
# c_best = max_score.parameters['C']
# print(c_best)

svm_clf = SVC(C=1.0, kernel='linear', random_state=241)
svm_clf.fit(X,y)

coef = svm_clf.coef_.toarray().ravel()
top_10 = np.argsort(np.abs(svm_clf.coef_.toarray()[0]))[-10:]
print(top_10)
words = [vectorizer.get_feature_names()[i] for i in top_10]
words.sort()
print(words)