import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

data = pd.read_csv('close_prices.csv')
cols = data.columns.values.tolist()
print(cols)
data = data.values[:, 1:]


pca = PCA(n_components=10)
pca.fit(data)
print('Explained variance ratio:')
print(pca.explained_variance_ratio_)

new_data = pd.DataFrame(pca.transform(data))
dj_data = pd.read_csv('djia_index.csv')

c = np.corrcoef(new_data[0], dj_data['^DJI'])[0,1]

print('Correlation between first component if the new data and Dj index:')
print('{0:.2f}'.format(c))

idx = np.argmax(pca.components_[0]) + 1
print(cols[idx])