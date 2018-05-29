
# main.py Perfomr PCA
# Cloud Cho May 26, 2018
#
# error exist

import numpy as np
import sklearn
from sklearn.decomposition import PCA

print(sklearn.__version__)

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca = PCA(n_components=2)
pca.fit(X)
print(pca.explained_variance_ratio_)  
# Error
# Cause: version under 0.19
#print(pca.singular_values_)  

pca = PCA(n_components=2, svd_solver='full')
pca.fit(X)                 
print(pca.explained_variance_ratio_)  
#print(pca.singular_values_)  

pca = PCA(n_components=1, svd_solver='arpack')
pca.fit(X)
print(pca.explained_variance_ratio_)  
#print(pca.singular_values_)  

