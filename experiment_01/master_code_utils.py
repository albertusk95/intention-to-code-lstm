
# File: master_code_utils.py
# This file contains the code utility functions

from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent, Embedding, Dropout
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam, RMSprop 
from nltk import FreqDist
import numpy as np
import os
import datetime

def loadData(source):
	
	# Reading data
	f = open(source, 'r')
	data = f.read()
	f.close()
	
	return data

def fetchNL(nl_raw_data):
	splittedNL = nl_raw_data.split('\n')
	return splittedNL

def fetchCode(code_raw_data, num_of_lines):
	splittedCode = code_raw_data.split('\n')
	
	# Combining some lines of code into one corresponding code
	tmp_code = ''
	listOfOneCode = []
	counter = 0
	for sc in splittedCode:
		if sc != '----- ':
			if counter < num_of_lines:
				tmp_code += sc
				counter += 1

		else:
			listOfOneCode.append(tmp_code)
			tmp_code = ''
			counter = 0

	return listOfOneCode

def fetchCodeAllLines(code_raw_data):
	splittedCode = code_raw_data.split('\n')
	
	# Combining some lines of code into one corresponding code
	tmp_code = ''
	listOfOneCode = []
	for sc in splittedCode:
		if sc != '----- ':
			tmp_code += sc

		else:
			listOfOneCode.append(tmp_code)
			tmp_code = ''

	return listOfOneCode

def reverseOrderOfListOfWords(list_of_list_of_words):
	listOfListOfWords = []
	for sc in list_of_list_of_words:
		listOfListOfWords.append(sc[::-1])

	return listOfListOfWords

def convertSentenceToSeqOfWords(list_of_input):
	listOfInputWords = []
	for sc in list_of_input:
		listOfInputWords.append(text_to_word_sequence(sc, filters='"#$;?@`~\t\n'))

	return listOfInputWords

def createVocabulary(list_of_list_of_words, vocab_size):
	fd = FreqDist(np.hstack(list_of_list_of_words))
	return fd.most_common(vocab_size-1)

def createIdxToWord(vocab_words):
	idx_to_word = [word[0] for word in vocab_words]
	idx_to_word.insert(0, 'ZERO')
	idx_to_word.append('UNK')
	return idx_to_word

def createWordToIdx(idx_to_word):
	word_to_idx = {word:idx for idx, word in enumerate(idx_to_word)}
	return word_to_idx

def convertListOfWordsToIdx(list_of_list_of_words, word_to_idx):
	for i, input_sentence in enumerate(list_of_list_of_words):
		for j, input_word in enumerate(input_sentence):
			if input_word in word_to_idx:
				list_of_list_of_words[i][j] = word_to_idx[input_word]
			else:
				list_of_list_of_words[i][j] = word_to_idx['UNK']

def computeMaxLen(list_of_list_of_words):
	return max([len(il) for il in list_of_list_of_words])

def paddingZeros(list_of_list_of_words, words_max_len):
	return pad_sequences(list_of_list_of_words, maxlen=words_max_len, dtype='int32', padding='post')

def createModel(nl_vocab_len, nl_max_len, code_vocab_len, code_max_len, hidden_dim, layer_num):
	
	model = Sequential()

	''' ENCODER '''
	# Using an Embedding layer
	model.add(Embedding(nl_vocab_len, 1000, input_length=nl_max_len, mask_zero=True))
	
	# Creating a single LSTM layer with hidden_dim as the number of units
	model.add(LSTM(hidden_dim))

	# Suppose the model.output_shape == (None, 32)
	# note: `None` is the batch dimension
	# And we do this: model.add(RepeatVector(3))
	# Now, the value of model.output_shape == (None, 3, 32)
	model.add(RepeatVector(code_max_len))

	''' DECODER '''

	# Creating LSTM layers with the total number of layer_num in which each layer has hidden_dim units
	for idx in range(layer_num):
		model.add(LSTM(hidden_dim, return_sequences=True))

	model.add(TimeDistributed(Dense(code_vocab_len)))
	model.add(Activation('softmax'))
	model.compile(loss='categorical_crossentropy',
				optimizer='adam',
				metrics=['accuracy'])
	
	return model
	
def vectorizeListOfWords(padded_list_of_words, max_len, word_to_idx):

	# Vectorizing each element in each sequence (create one-hot vector)
	sequences = np.zeros((len(padded_list_of_words), max_len, len(word_to_idx)))
	for i, sentence in enumerate(padded_list_of_words):
		for j, word in enumerate(sentence):
			sequences[i, j, word] = 1.
	return sequences

def findCheckpointFile(file_dir):
	
	# Returning the latest built model based on the modified time
	checkpointFile = [f for f in os.listdir(file_dir) if 'checkpoint' in f]
	
	if len(checkpointFile) == 0:
		return []
	
	modifiedTime = [os.path.getmtime('model_for_testing/'+f) for f in checkpointFile]
	
	return checkpointFile[np.argmax(modifiedTime)]

