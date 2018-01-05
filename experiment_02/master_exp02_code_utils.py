
# File: master_exp02_utils.py
# This file contains all methods for supporting the main controller

import nltk
import numpy as np
import operator
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn

#nltk.download('averaged_perceptron_tagger')

def loadData(source):
	
	# Reading data
	f = open(source, 'r')
	data = f.read()
	f.close()
	
	return data

def fetchNL(nl_raw_data):
	splittedNL = nl_raw_data.split('\n')
	return splittedNL

def fetchCode(code_raw_data):
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

def getRandIdxOfStructuredText(baseProbMinIdx, baseProbMaxIdx):
	return np.random.randint(baseProbMinIdx, baseProbMaxIdx-1)
