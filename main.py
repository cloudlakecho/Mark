
# main.py in Anthony-Stockholm 
# Cloud Cho May 26, 2018 - Perfomr Pricipal Component Analysis, PCA
#
# Error:
#   error exist
# 
# Comment:
#

import numpy as np
import sklearn
from sklearn.decomposition import PCA

import argparse
import keras
import os.path

from pathlib2 import Path #  pipy library
from inspect import currentframe, getframeinfo
from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten, Conv2D


def main():
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    pca = PCA(n_components=2)
    pca.fit(X)
    print(pca.explained_variance_ratio_)  
    
    # Error spot
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


if __name__ == '__main__':
    print(sklearn.__version__)
    main()
    
