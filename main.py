
# main.py in Mark
# @author Cloud Cho December 7, 2018 - To generate story
#
# Error:
#
# To do: 
#	add some in train_model function
#      
# Comment:
#
#

import numpy as np
import sklearn
from sklearn.decomposition import PCA

import argparse
import keras
print("Keras impoted")
import os.path

# from pathlib2 import Path #  pipy library
from inspect import currentframe, getframeinfo
from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten, Conv2D

from tool import lstm


#
# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
#
def main():
	file_name = raw_input("Please, give input text file, thanks.")
	n_char, n_vocab, raw_text = lstm.data_manipulation(file_name)
	X, y = make_dataset(n_chars, n_vocab, raw_text)
	train_model(X, y)


if __name__ == '__main__':
    main()
