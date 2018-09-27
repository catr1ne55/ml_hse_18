import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier


gender = {'male': 1,'female': 0}
df = pd.read_csv('titanic.csv', index_col='PassengerId')[['Pclass', 'Fare', 'Age', 'Sex', 'Survived']]
df['Sex'] = np.where(df['Sex'] == 'male',1,0)
df = df.dropna()

X = df[['Pclass', 'Fare', 'Age', 'Sex']]
y = df['Survived']

clf = DecisionTreeClassifier(random_state=241)
clf.fit(X, y)

importances = clf.feature_importances_
print(list(zip(X.columns, importances)))