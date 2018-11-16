from sklearn.svm import SVC
import pandas as pd

svm = SVC(C=100000, kernel='linear', random_state=241)

data = pd.read_csv('svm-data.csv', header=None)
data.columns = ['y', 'x1', 'x2' ]
X = data[['x1', 'x2']]
y = data[['y']]

svm.fit(X,y)
svectors = svm.support_
print(svectors)