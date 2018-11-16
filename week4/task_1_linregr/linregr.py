import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from scipy.sparse import hstack

df = pd.read_csv('salary-train.csv')
df = df.applymap(lambda s:s.lower() if type(s) == str else s)
df['FullDescription'] = df['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)

vectorizer = TfidfVectorizer(min_df=5)
texts = vectorizer.fit_transform(df['FullDescription'])

df['LocationNormalized'].fillna('nan', inplace=True)
df['ContractTime'].fillna('nan', inplace=True)
enc = DictVectorizer()
categ = enc.fit_transform(df[['LocationNormalized', 'ContractTime']].to_dict('records'))

X = hstack([texts, categ])
y = df['SalaryNormalized']

test = pd.read_csv('salary-test-mini.csv')
test = test.applymap(lambda s:s.lower() if type(s) == str else s)
test['FullDescription'] = test['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)

texts_test = vectorizer.transform(test['FullDescription'])

test['LocationNormalized'].fillna('nan', inplace=True)
test['ContractTime'].fillna('nan', inplace=True)
categ_test = enc.transform(test[['LocationNormalized', 'ContractTime']].to_dict('records'))

X_test = hstack([texts_test, categ_test])


ridge_clf = Ridge(alpha=1, random_state=241)
ridge_clf.fit(X, y)
print(ridge_clf.predict(X_test))