import pandas as pd
import scipy.stats as st

data = pd.read_csv('titanic.csv', index_col='PassengerId')

sibsp = data.SibSp.value_counts()
parch = data.Parch.value_counts()
print(data.corr(method='pearson'))