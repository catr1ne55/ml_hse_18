import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score



data = pd.read_csv('abalone.csv')
data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))
y = data['Rings']
X = data.drop('Rings', axis=1)

gen = KFold(n_splits=5, random_state=1, shuffle=True)

for i in range(1, 50):
    clf = RandomForestRegressor(n_estimators=i, random_state=1)
    score = cross_val_score(clf, X, y, cv=gen, scoring='r2').mean()
    if score > 0.52:
        print('Min number of trees is ', i)
        break