
# File: master_exp02_train.py
# This file contains all methods for fitting the network model

from master_exp02_code_utils import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import argparse
import numpy

# Input parameters
param = argparse.ArgumentParser()
param.add_argument('-hiddendim', default=256)
param.add_argument('-epoch', default=40)
param.add_argument('-batchsize', default=128)
param.add_argument('-seqlen', default=60)
params = vars(param.parse_args())

HIDDEN_DIM = params['hiddendim']
NB_EPOCH = params['epoch']
BATCH_SIZE = params['batchsize']
SEQ_LEN = params['seqlen']

# List of training data locations
listOfCodeTrainingDataLocations = [
									'../../data/training/prob2/JOHNY_CODE_ORIGIN',
									'../../data/training/prob3/HORSES_CODE_ORIGIN',
									'../../data/training/prob4/STONES_CODE_ORIGIN'
									]


# Introduction
print '\n'
print 'Welcome to the 2nd Experiment --- Text Generator Model Training'
print '-------------------------------------------------------------------------'
print 'Topic: Source Code Generation Based On User Intention Using LSTM Networks'
print 'By: Albertus Kelvin'
print '-------------------------------------------------------------------------'
print '\n'


# Some Network Information
print 'Parameters information:'
print ('- lstm units: {}'.format(HIDDEN_DIM))
print ('- epochs: {}'.format(NB_EPOCH))
print ('- batch size: {}'.format(BATCH_SIZE))
print ('- number of sequences: {}'.format(SEQ_LEN))
print '\n'


# Loading training data
print '[INFO] Loading Code training data...'

codeTrainingData = ''
for train_data_loc in listOfCodeTrainingDataLocations:
	codeTrainingData += loadData(train_data_loc)
	codeTrainingData += '\n'

print codeTrainingData
print '[INFO] Done\n'


# Converting to lower case
print '[INFO] Converting Code training data to lower case...'

codeTrainingData = codeTrainingData.lower()

print '[INFO] Done\n'


# Creating mapping of unique chars to integers
print '[INFO] Creating mapping of unique chars to integers...'

chars = sorted(list(set(codeTrainingData)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

print '[INFO] Done\n'


# Summarizing the loaded data
print '[INFO] Computing the length of chars and vocab...'

n_chars = len(codeTrainingData)
n_vocab = len(chars)

print ('Total Characters: {}'.format(n_chars))
print ('Total Vocab: {}'.format(n_vocab))
print '[INFO] Done\n'


# Preparing the dataset of input to output pairs encoded as integers
seq_len = SEQ_LEN
dataX = []
dataY = []

print '[INFO] Building the training data...'

for i in range(0, n_chars - seq_len, 1):
	seq_in = codeTrainingData[i:i + seq_len]
	seq_out = codeTrainingData[i + seq_len]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])

print '[INFO] Done\n'


print '[INFO] Computing total patterns...'

n_patterns = len(dataX)

print ('Total Patterns: {}'.format(n_patterns))
print '[INFO] Done\n'


# Reshape X to be [samples, time steps, features]
print '[INFO] Reshaping the input samples...'

X = numpy.reshape(dataX, (n_patterns, seq_len, 1))

print '[INFO] Done\n'


# Normalizing
print '[INFO] Normalizing the input samples...'

X = X / float(n_vocab)

print '[INFO] Done\n'


# One hot encode the output variable
print '[INFO] Vectorizing the output samples...'

y = np_utils.to_categorical(dataY)

print '[INFO] Done\n'


# Defining the LSTM model
print '[INFO] Building the network model...'

# define the LSTM model
'''
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
#model.compile(loss='categorical_crossentropy', optimizer='adam')

model.compile(loss='categorical_crossentropy',
				optimizer='adam',
				metrics=['accuracy'])

# define the checkpoint
filepath="weights-improvement-1l256u{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# fit the model
model.fit(X, y, epochs=NB_EPOCH, batch_size=BATCH_SIZE, callbacks=callbacks_list)
'''

'''
model = Sequential()
model.add(LSTM(HIDDEN_DIM, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(HIDDEN_DIM))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

#model.compile(loss='categorical_crossentropy', optimizer='adam')

model.compile(loss='categorical_crossentropy',
				optimizer='adam',
				metrics=['accuracy'])

print '[INFO] Building network model done...'
print model.summary()

# Defining the checkpoint
print '[INFO] Starting to train the network model...'

filepath="models/master-exp02-2l1000u{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# Fitting the model
model.fit(X, y, epochs=NB_EPOCH, batch_size=BATCH_SIZE, callbacks=callbacks_list)
'''


model = Sequential()
model.add(LSTM(HIDDEN_DIM, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(HIDDEN_DIM))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

#model.compile(loss='categorical_crossentropy', optimizer='adam')

model.compile(loss='categorical_crossentropy',
				optimizer='adam',
				metrics=['accuracy'])

print model.summary()
print '[INFO] Done\n'


# Defining the checkpoint
print '[INFO] Starting to train the network model...'

# If any trained weight was found, then load them into the model
model.load_weights('model_for_testing/master-3types-2l256u-next2-50-0.2361.hdf5')


filepath="models/master-exp02-2l256u60s-final-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# Fitting the model
model.fit(X, y, epochs=NB_EPOCH, batch_size=BATCH_SIZE, callbacks=callbacks_list)

print '[INFO] Done\n'
