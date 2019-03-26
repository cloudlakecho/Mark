
# main.py in Mark
# Cloud Cho December 7, 2018 - To generate story
#
# Error:
#
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

from pathlib2 import Path #  pipy library
from inspect import currentframe, getframeinfo
from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten, Conv2D

from tool import lstm


#
# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
#
def main():
	file_name = raw_input("Please, give input text file, thanks.")
	lstm.data_manipulation(file_name)
	make_dataset()
	train_model()


if __name__ == '__main__':
    main()
