
# lstm.py in Benjamin project
# Clouc Cho May 6, 2018 ~ - Generate text from input story
#
# Reference:
#   https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/
# 	Small LSTM Network to Generate Text for Alice in Wonderland
#

# Hot wo run this code
# (1) $ python lstm.py
# (2) Give TXT file location
#   ex) ../../../../wuthering_heights.txt
#
# Work? - not yet?
#
#

import pdb
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils


EARLY_DEBUGGING = False
DEBUGGING = True
TESTING = True


def data_manipulation(file_name):
    # load ascii text and covert to lowercase
	raw_text = open(file_name).read()
	raw_text = raw_text.lower()
	# create mapping of unique chars to integers
	chars = sorted(list(set(raw_text)))
	char_to_int = dict((c, i) for i, c in enumerate(chars))
	# summarize the loaded data
	n_chars = len(raw_text)
	n_vocab = len(chars)
	print ("Total Characters: ", n_chars)
	print ("Total Vocab: ", n_vocab)

	return n_chars, n_vocab, raw_text, char_to_int


def make_dataset(n_chars, n_vocab, raw_text, char_to_int, seq_length):
	# prepare the dataset of input to output pairs encoded as integers
	if (seq_lenght == None):
		seq_length = 100

	dataX = []
	dataY = []
	for i in range(0, n_chars - seq_length, 1):
		seq_in = raw_text[i:i + seq_length]
		seq_out = raw_text[i + seq_length]
		dataX.append([char_to_int[char] for char in seq_in])
		dataY.append(char_to_int[seq_out])
	n_patterns = len(dataX)
	print ("Total Patterns: ", n_patterns)
	# reshape X to be [samples, time steps, features]
	X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
	# normalize
	X = X / float(n_vocab)
	# one hot encode the output variable
	y = np_utils.to_categorical(dataY)

	if (TESTING):
		pdb.set_trace()

	return X, y


def train_model(X, y, seq_len=256, drop_out=0.2):
	# define the LSTM model
	model = Sequential()

	# To do
	# Sequence length effect on LSTM design?
	model.add(LSTM(seq_len, input_shape=(X.shape[1], X.shape[2])))

	model.add(Dropout(drop_out))
	model.add(Dense(y.shape[1], activation='softmax'))

	# To do
	# How about on more LSTM and activation?
	if (DEBUGGING):
		pdb.set_trace()

	model.compile(loss='categorical_crossentropy', optimizer='adam')
	# define the checkpoint
	filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"

	# To do
	# Not save training record
	checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1,
		save_best_only=True, mode='min')
	callbacks_list = [checkpoint]

	# fit the model
	model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)

	if (DEBUGGING):
		pdb.set_trace()



#
# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
#
def main():
	file_name = input("Please, give input text file, thanks.")
	n_chars, n_vocab, raw_text, char_to_int = data_manipulation(file_name)
	X, y = make_dataset(n_chars, n_vocab, raw_text, char_to_int)
	train_model(X, y)


if __name__ == '__main__':
    main()
