
# main.py in Mark
# @author Cloud Cho December 7, 2018 - To generate story
#
# Commad line:
# 	python main.py /home/cloud/Documents/wuthering_heights.txt
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

import keras
print("Keras impoted")
import os.path

# from pathlib2 import Path #  pipy library
from inspect import currentframe, getframeinfo
from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten, Conv2D

from tool import lstm

import argparse
import sys


#
# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
#
def main():
	if (len(sys.argv) != 2):
		print('Please, give input story in TEXT format, thanks.')
		print('Current no. of inputs: %d' % len(sys.argv))
		sys.exit()
	else:
		file_name = sys.argv[1]

	n_chars, n_vocab, raw_text, char_to_int = lstm.data_manipulation(file_name)
	print('Step 1: Story read')
	X, y = lstm.make_dataset(n_chars, n_vocab, raw_text, char_to_int)
	print('Step 2: Dataset generated')
	lstm.train_model(X, y)
	print('Step 3: Neural network trained')


if __name__ == '__main__':
    main()
