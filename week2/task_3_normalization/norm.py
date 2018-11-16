from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

train_data = pd.read_csv('perceptron-train.csv', header=None)
train_data.columns = ['y', 'x1', 'x2' ]
test_data = pd.read_csv('perceptron-test.csv', header=None)
test_data.columns = ['y', 'x1', 'x2' ]

scaler = StandardScaler()
X_train = train_data[['x1', 'x2']]
y_train = train_data[['y']]

X_test = test_data[['x1', 'x2']]
y_test = test_data[['y']]

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf1 = Perceptron(random_state=241)
clf1.fit(X_train, y_train)
predictions = clf1.predict(X_test)
acc = accuracy_score(y_test, predictions)

clf2 = Perceptron(random_state=241)
clf2.fit(X_train_scaled, y_train)
predictions_scaled = clf2.predict(X_test_scaled)
acc_scaled = accuracy_score(y_test, predictions_scaled)

print(abs(acc - acc_scaled))

