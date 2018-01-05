
# File: master_exp02_test.py
# This file contains script for generating the final source code

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import argparse
import numpy
import sys

# Input parameters
param = argparse.ArgumentParser()
param.add_argument('-hiddendim', default=256)
param.add_argument('-seqlen', default=100)
params = vars(param.parse_args())

HIDDEN_DIM = params['hiddendim']
SEQ_LEN = params['seqlen']



def convertToLowerCase(code_training_data):
	# Converting to lower case
	return code_training_data.lower()

def createMappingCharToInt(code_training_data):
	# Creating mapping of unique chars to integers, and a reverse mapping
	chars = sorted(list(set(code_training_data)))
	char_to_int = dict((c, i) for i, c in enumerate(chars))
	return char_to_int

def createMappingIntToChar(code_training_data):
	# Creating mapping of integers to unique char
	chars = sorted(list(set(code_training_data)))
	int_to_char = dict((i, c) for i, c in enumerate(chars))
	return int_to_char

def computeTotalChars(code_training_data):
	n_chars = len(code_training_data)
	print ('Total Characters: {}'.format(n_chars))
	return n_chars

def computeTotalVocab(code_training_data):
	chars = sorted(list(set(code_training_data)))
	n_vocab = len(chars)
	print ('Total Vocab: {}'.format(n_vocab))
	return n_vocab

def createIOPairs(code_training_data, n_chars, char_to_int):
	# Preparing the dataset of input to output pairs encoded as integers
	seq_length = SEQ_LEN
	dataX = []
	dataY = []

	for i in range(0, n_chars - seq_length, 1):
		seq_in = code_training_data[i:i + seq_length]
		seq_out = code_training_data[i + seq_length]
		dataX.append([char_to_int[char] for char in seq_in])
		dataY.append(char_to_int[seq_out])

	return dataX, dataY

def computeTotalPatterns(data_x):
	n_patterns = len(data_x)
	
	print '[INFO] Computing total patterns done...'
	print ('Total Patterns: {}'.format(n_patterns))

	return n_patterns

def reshapeAndNormalize(data_x, n_patterns, n_vocab):
	# Reshaping X to be [samples, time steps, features]
	X = numpy.reshape(data_x, (n_patterns, SEQ_LEN, 1))

	# Normalizing
	X = X / float(n_vocab)

	return X

def createOneHotVector(data_y):
	# One hot encode the output variable
	y = np_utils.to_categorical(data_y)

	return y

def loadModel(x, y):
	# Defining the LSTM model
	print '[INFO] Starting to build the network model...'

	'''
	model = Sequential()
	model.add(LSTM(256, input_shape=(x.shape[1], x.shape[2])))
	model.add(Dropout(0.2))
	model.add(Dense(y.shape[1], activation='softmax'))
	#model.compile(loss='categorical_crossentropy', optimizer='adam')

	model.compile(loss='categorical_crossentropy',
					optimizer='adam',
					metrics=['accuracy'])
	'''

	
	model = Sequential()
	model.add(LSTM(HIDDEN_DIM, input_shape=(x.shape[1], x.shape[2]), return_sequences=True))
	model.add(Dropout(0.2))
	model.add(LSTM(HIDDEN_DIM))
	model.add(Dropout(0.2))
	model.add(Dense(y.shape[1], activation='softmax'))
	


	# Loading the network weights
	#filename = "model_for_testing/master-3types-2l256u-next2-50-0.2361.hdf5"
	filename = "model_for_testing/master-3types100s-2l256u-next107-0.2996.hdf5"
	model.load_weights(filename)
	model.compile(loss='categorical_crossentropy', optimizer='adam')

	print '[INFO] Building network model done...'
	print model.summary()

	return model