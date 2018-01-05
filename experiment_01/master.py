
# File: master.py
# This file is the main controller of the entire program

from master_nl_utils import *
from master_code_utils import *
from master_textgen_test import *
import numpy as np
import argparse
import operator
import sys
from keras.callbacks import ModelCheckpoint

# Input parameters
param = argparse.ArgumentParser()
param.add_argument('-mode', default='test')
param.add_argument('-vocabsize', default=100)
param.add_argument('-hiddendim', default=1000)
param.add_argument('-layernum', default=4)
param.add_argument('-epoch', default=60)
param.add_argument('-batchsize', default=10)
param.add_argument('-numoflines', default=9)
param.add_argument('-seqlen', default=100)
params = vars(param.parse_args())

MODE = params['mode']
VOCAB_SIZE = params['vocabsize']
HIDDEN_DIM = params['hiddendim']
LAYER_NUM = params['layernum']
NB_EPOCH = params['epoch']
BATCH_SIZE = params['batchsize']
NUM_OF_LINES = params['numoflines']
SEQ_LEN = params['seqlen']


# Location of training data
listOfNLTrainingDataLocations = [
									'../../data/training/prob2/JOHNY_NL',
									'../../data/training/prob3/HORSES_NL',
									'../../data/training/prob4/STONES_NL'
								]


listOfCodeTrainingDataLocations = [
									'../../data/training/prob2/JOHNY_CODE_WORD',
									'../../data/training/prob3/HORSES_CODE_WORD',
									'../../data/training/prob4/STONES_CODE_WORD'
									]


listOfOriginCodeTrainingDataLocations = [
									'../../data/training/prob2/JOHNY_CODE_ORIGIN',
									'../../data/training/prob3/HORSES_CODE_ORIGIN',
									'../../data/training/prob4/STONES_CODE_ORIGIN'
									]


# List of base problem statements
listOfBaseProbStatements = [
    							'find the position of "uncle johny" in the sorted playlist',
    							'report the minimum difference that is possible between two horses in the race',
    							'you have to find out how many characters of s are in j as well'
							]


# List of structured format of base problem statements
listOfStructuredBaseProbStatements = [
								'find string position array index',
								'calculate minimum diff array difference',
								'find same chars two str total'
							]


# Introduction
print '\n'
print 'Welcome to the 1st Experiment --- Main Program'
print '-------------------------------------------------------------------------'
print 'Topic: Source Code Generation Based On User Intention Using LSTM Networks'
print 'By: Albertus Kelvin'
print '-------------------------------------------------------------------------'
print '\n'


# Some Network Information
print 'Parameters information:'
print ('- mode: {}'.format(MODE))
print ('- vocab size: {}'.format(VOCAB_SIZE))
print ('- lstm units: {}'.format(HIDDEN_DIM))
print ('- lstm layers: {}'.format(LAYER_NUM))
print ('- epochs: {}'.format(NB_EPOCH))
print ('- batch size: {}'.format(BATCH_SIZE))
print ('- number of lines: {}'.format(NUM_OF_LINES))
print ('- number of sequences: {}'.format(SEQ_LEN))
print '\n'


# Get the number of generated code chars from the user
print 'Please read these rules:'
print '1. You can specify how the code generation process stops'
print '2. The possible choices are:'
print '   - specifying a number of generated characters'
print '   - using the string "-----"'
print '3. If you choose to use the string "-----", please give -1 as the input below\n'

numOfGeneratedCodeChars = int(raw_input('Number of generated code chars: '))


# Loading training data
print '[INFO] Loading NL training data...'

nlTrainingData = ''
for train_data_loc in listOfNLTrainingDataLocations:
	nlTrainingData += loadData(train_data_loc)

print '[INFO] Done\n'


print '[INFO] Loading Code training data...'

codeTrainingData = ''
for train_data_loc in listOfCodeTrainingDataLocations:
	codeTrainingData += loadData(train_data_loc)
	codeTrainingData += '\n'

print '[INFO] Done\n'


# Splitting raw text into array of sequences
print '[INFO] Splitting raw text of NL to array of sentences...'

listOfNL = fetchNL(nlTrainingData)

print '[INFO] Done\n'


# Filtering the empty space in the list of NL
print '[INFO] Filtering out the empty space in the list of NL'

listOfNL = filter(lambda lon: lon != '', listOfNL)

print ('Total NL training data: {}'.format(len(listOfNL)))
print '[INFO] Done\n'


print '[INFO] Splitting raw text of Code to array of sentences...'

listOfCodes = fetchCode(codeTrainingData, NUM_OF_LINES)

print ('Total Code training data: {}'.format(len(listOfCodes)))
print '[INFO] Done\n'


# Converting each sentence in the list into sequence of words
print '[INFO] Converting each of NL sentence into sequence of words...'

listOfListOfNLWords = convertSentenceToSeqOfWords(listOfNL)

print ('Total list of NL words: {}'.format(len(listOfListOfNLWords)))
print '[INFO] Done\n'


print '[INFO] Reversing the order of list of NL words...'

listOfReversedOrderOfListOfNLWords = reverseOrderOfListOfWords(listOfListOfNLWords)

print ('Total list of NL words in reversed order: {}'.format(len(listOfReversedOrderOfListOfNLWords)))
print '[INFO] Done\n'


print '[INFO] Converting each of Code sentence into sequence of words...'

listOfListOfCodeWords = convertSentenceToSeqOfWords(listOfCodes)

print ('Total list of Code words: {}'.format(len(listOfListOfCodeWords)))
print '[INFO] Done\n'


# Creating the vocabulary
print '[INFO] Creating NL vocabulary...'

nlVocab = createVocabulary(listOfReversedOrderOfListOfNLWords, VOCAB_SIZE)

print ('Total NL vocabulary: {}'.format(len(nlVocab)))
print '[INFO] Done\n'


print '[INFO] Creating Code vocabulary done...'

codeVocab = createVocabulary(listOfListOfCodeWords, VOCAB_SIZE)

print ('Total Code vocabulary: {}'.format(len(codeVocab)))
print '[INFO] Done\n'


# Creating dictionary for index-to-word
print '[INFO] Creating NL index to word...'

nlIdxToWord = createIdxToWord(nlVocab)

print ('Total NL index to word: {}'.format(len(nlIdxToWord)))
print '[INFO] Done\n'


print '[INFO] Creating Code index to word...'

codeIdxToWord = createIdxToWord(codeVocab)

print ('Total Code index to word: {}'.format(len(codeIdxToWord)))
print '[INFO] Done\n'


# Creating dictionary for word-to-index
print '[INFO] Creating NL word to index...'

nlWordToIdx = createWordToIdx(nlIdxToWord)

print ('Total NL word to index: {}'.format(len(nlWordToIdx)))
print '[INFO] Done\n'


print '[INFO] Creating Code word to index...'

codeWordToIdx = createWordToIdx(codeIdxToWord)

print ('Total Code word to index: {}'.format(len(codeWordToIdx)))
print '[INFO] Done\n'


# Converting list of words (training data) to the corresponding index
print '[INFO] Converting list of reversed NL words to index...'

convertListOfWordsToIdx(listOfReversedOrderOfListOfNLWords, nlWordToIdx)

print ('Total converted list of reversed NL words to index: {}'.format(len(listOfReversedOrderOfListOfNLWords)))
print '[INFO] Done\n'


print '[INFO] Converting list of Code words to index...'

convertListOfWordsToIdx(listOfListOfCodeWords, codeWordToIdx)

print ('Total converted list of Code words to index: {}'.format(len(listOfListOfCodeWords)))
print '[INFO] Done\n'


# Compute the maximum length of list of words
print '[INFO] Computing the maximum length of list of list of NL words...'

maxLenOfNL = computeMaxLen(listOfListOfNLWords)

print maxLenOfNL
print ('Total computed list of NL words: {}'.format(len(listOfListOfNLWords)))
print '[INFO] Done\n'


print '[INFO] Computing the maximum length of list of list of Code words...'

maxLenOfCode = computeMaxLen(listOfListOfCodeWords)

print maxLenOfCode
print ('Total computed list of Code words: {}'.format(len(listOfListOfCodeWords)))
print '[INFO] Done\n'


# Padding zero so all the sequence of words have the same length as the longest one
print '[INFO] Padding zeros for list of reversed NL words...'

paddedListOfNLWords = paddingZeros(listOfReversedOrderOfListOfNLWords, maxLenOfNL)

print ('Total list of zero padded for NL words: {}'.format(len(paddedListOfNLWords)))
print '[INFO] Done\n'


print '[INFO] Padding zeros for list of Code words...'

paddedListOfCodeWords = paddingZeros(listOfListOfCodeWords, maxLenOfCode)

print ('Total list of zero padded for Code words: {}'.format(len(paddedListOfCodeWords)))
print '[INFO] Done\n'


# Compute the lenght of vocabulary by including the ZERO and UNK token
lenOfNLVocabWithZEROAndUNK = len(nlVocab)+2
lenOfCodeVocabWithZEROAndUNK = len(codeVocab)+2

# Creating the network model
print '[INFO] Building the network model...'

networkModel = createModel(lenOfNLVocabWithZEROAndUNK, maxLenOfNL, lenOfCodeVocabWithZEROAndUNK, maxLenOfCode, HIDDEN_DIM, LAYER_NUM)
networkModel.summary()

print '[INFO] Done\n'


# Searching for the saved model
savedModel = findCheckpointFile('model_for_testing/')

# Start the training process
if MODE == 'train':
	
	
	# If any trained weight was found, then load them into the model
	if len(savedModel) != 0:
		print('[INFO] Saved model found, loading...')
		
		networkModel.load_weights('model_for_testing/'+savedModel)
		
		print '[INFO] Done\n'


	# Vectorization
	print '[INFO] Creating one-hot vector for Code samples...'

	vectorizedListOfCodeWords = vectorizeListOfWords(paddedListOfCodeWords, maxLenOfCode, codeWordToIdx)

	print '[INFO] Done\n'


	# Defining the checkpoint
	print '[INFO] Starting to train the network model...'

	filepath = "models/model-9lines3typesfinalv01-{epoch:02d}-{loss:.4f}.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
	callbacks_list = [checkpoint]

	# fit the model
	networkModel.fit(paddedListOfNLWords, vectorizedListOfCodeWords, epochs=NB_EPOCH, batch_size=BATCH_SIZE, callbacks=callbacks_list)

	print '[INFO] Done\n'


else:
	
	
	print '[INFO] ENCODER-DECODER MODEL'
	print '----------------------------\n'

	print '[INFO] Searching for any saved Encoder-Decoder model...'


	# Checking whether there is saved model
	if len(savedModel) == 0:
		print '[INFO] Could not find any Encoder-Decoder model...'
		sys.exit()
	else:
		
		print '[INFO] Encoder-Decoder model found...'


		# Loading the saved model
		print '[INFO] Loading Encoder-Decoder network model...'

		networkModel.load_weights('model_for_testing/'+savedModel)

		print networkModel.summary()
		print '[INFO] Done\n'


		# TEXT GENERATOR INITIALIZATION
		print '[INFO] TEXT GENERATOR MODEL'
		print '---------------------------\n'


		
		print '[INFO] Loading Code training data...'

		codeTrainingDataOrigin = ''
		for train_data_loc in listOfOriginCodeTrainingDataLocations:
			codeTrainingDataOrigin += loadData(train_data_loc)
			codeTrainingDataOrigin += '\n'

		print '[INFO] Done\n'
		

		print '[INFO] Converting training data to lower case...'
		
		codeTrainingDataOrigin = convertToLowerCase(codeTrainingDataOrigin)
		
		print '[INFO] Done\n'
		

		print '[INFO] Creating mapping for char to int...'
		
		charToInt = createMappingCharToInt(codeTrainingDataOrigin)
		
		print '[INFO] Done\n'

		
		print '[INFO] Creating mapping for int to char...'
		
		intToChar = createMappingIntToChar(codeTrainingDataOrigin)
		
		print '[INFO] Done\n'


		print '[INFO] Computing the number of chars...'
		
		numOfChars = computeTotalChars(codeTrainingDataOrigin)
		
		print '[INFO] Done\n'


		print '[INFO] Computing the number of vocab...'
		
		numOfVocab = computeTotalVocab(codeTrainingDataOrigin)
		
		print '[INFO] Done\n'


		print '[INFO] Creating IO pairs...'
		
		dataX, dataY = createIOPairs(codeTrainingDataOrigin, numOfChars, charToInt)
		
		print '[INFO] Done\n'


		print '[INFO] Computing the number of patterns...'
		
		numOfPatterns = computeTotalPatterns(dataX)
		
		print '[INFO] Done\n'


		print '[INFO] Reshaping and normalizing input pattern...'
		
		reshapedAndNormalizedInput = reshapeAndNormalize(dataX, numOfPatterns, numOfVocab)
		
		print '[INFO] Done\n'


		print '[INFO] Vectorizing the output pattern...'
		
		vectorizedOutput = createOneHotVector(dataY)
		
		print '[INFO] Done\n'


		print '[INFO] Loading model for generating code...'
		
		textGenModel = loadModel(reshapedAndNormalizedInput, vectorizedOutput)
		
		print '[INFO] Done\n'



		idxOfBaseProbForTestingData = []


		print '[INFO] TESTING DATA PREPROCESSING'
		print '---------------------------------\n'


		# Loading testing data (raw text format)
		print '[INFO] Loading testing data...'
		
		testingData = loadData('../../data/testing/testing_data')
		
		print '[INFO] Done\n'


		# Fetching samples of testing data
		print'[INFO] Splitting testing raw text into list of sentences...'
		
		listOfTestingData = fetchNL(testingData)
		
		listOfTestingData = filter(lambda lotd: lotd != '', listOfTestingData)

		print ('Total testing data: {}'.format(len(listOfTestingData)))
		
		for lotd in listOfTestingData:
			print ('- {}'.format(lotd))

		print '[INFO] Done\n'

		print '\n'

		for td in listOfTestingData:

			idx, val = max(enumerate([computeCosineSimilarity(td, bp) for bp in listOfBaseProbStatements]), key=operator.itemgetter(1))
			print '[INFO] Computing similarity score...'
			print('Problem: {}'.format(td))
			print('Most similar with base number: {}'.format(idx))
			print('Base problem: {}'.format(listOfBaseProbStatements[idx]))
			print('Similarity score: {}'.format(val))
			print('Structured format: {}'.format(listOfStructuredBaseProbStatements[idx]))
			print '[INFO] Done\n'


			print '[INFO] Inserting the structured text to list of testing data...'

			listOfNLTestingData = [listOfStructuredBaseProbStatements[idx]]
			
			print('List of testing data: {}'.format(listOfNLTestingData))
			print '[INFO] Done\n'


			# MANUAL TESTING 

			#listOfNLTestingData = ['calculate minimum diff array difference']
			#listOfNLTestingData = ['find string position array index']
			#listOfNLTestingData = ['find same chars two str total']

			#print ('Manual testing: {}'.format(listOfNLTestingData))


			print '[INFO] Converting sentences to sequence of words...'

			listOfTestingDataWords = convertSentenceToSeqOfWords(listOfNLTestingData)
			
			print '[INFO] Done'


			# Reversing the order of list of testing data chars
			print '[INFO] Reversing the order of list of testing data words...'

			reversedOrderOfListOfTestingDataWords = reverseOrderOfListOfWords(listOfTestingDataWords)
			
			print '[INFO] Done'

			
			# Converting each char to the corresponding index value
			print '[INFO] Converting each word to the corresponding index value...'

			convertListOfWordsToIdx(reversedOrderOfListOfTestingDataWords, nlWordToIdx)
		 	
		 	print '[INFO] Done'

			
		 	# Padding zeros to the list of index of testing data chars
			print '[INFO] Padding zeros to the list of index of testing data words...'

			paddedListOfTestingDataIdxOfWords = paddingZeros(reversedOrderOfListOfTestingDataWords, maxLenOfNL)
			
			print '[INFO] Done'


			# Predicting the class
			print '[INFO] Starting to predict the class...'

			# Output shape of model prediction is [1, 703, 80]
			predictions = np.argmax(networkModel.predict(paddedListOfTestingDataIdxOfWords, verbose=1), axis=2)
			
			print '[INFO] Done'


			print '[INFO] Inserting the predicted words to list of sequences...'

			sequences = []
			for prediction in predictions:
				sequence = ' '.join([codeIdxToWord[index] for index in prediction if index > 0])
				print(sequence)
				sequences.append(sequence)
			
			# Postprocessing the output from the decoder
			print 'List of predicted words:'
			print sequences

			print '[INFO] Done\n'
				

			print '[INFO] Converting seed (pattern) to lower case...'

			# Converting seed to lower case
			str_pattern = convertToLowerCase(sequences[0])

			print 'Lowered-case pattern:'
			print str_pattern
			print '[INFO] Done\n'

			
			# Taking some last characters
			print '[INFO] Taking some last characters ...'

			if len(str_pattern) < SEQ_LEN:
				delta = SEQ_LEN - len(str_pattern)
				str_pattern = '?'*delta + str_pattern
			else:
				# Get the last SEQ_LEN characters
				delta = len(str_pattern) - SEQ_LEN
				str_pattern = str_pattern[delta:]

			print str_pattern
			print '[INFO] Done\n'

			
			# Converting pattern chars to index
			print '[INFO] Converting pattern chars to index...'
			
			pattern = [charToInt[idx] for idx in str_pattern]
			
			print pattern
			print '[INFO] Done\n'


			# Generating characters
			print '[INFO] Generating remaining codes...'

			generatedSequences = []

			
			# Checking the number of generated code by the Encoder-Decoder model
			if numOfGeneratedCodeChars != -1:

				# using the specified number of generated chars
				numOfGenCharsByEDModel = len(sequences[0])

				if numOfGenCharsByEDModel < numOfGeneratedCodeChars:
					remainingNumOfGenChars = numOfGeneratedCodeChars - numOfGenCharsByEDModel
				
					for i in range(remainingNumOfGenChars):
						x = np.reshape(pattern, (1, len(pattern), 1))
						x = x / float(numOfVocab)
						prediction = textGenModel.predict(x, verbose=0)
						index = np.argmax(prediction)
						result = intToChar[index]
						seq_in = [intToChar[value] for value in pattern]
						#sys.stdout.write(result)
						
						# PRINT AND APPEND
						print result
						generatedSequences.append(result)

						pattern.append(index)
						pattern = pattern[1:len(pattern)]


				else:
					if numOfGenCharsByEDModel > numOfGeneratedCodeChars:
						print 'The number of generated code is more than your input number'
					else:
						print 'The number of generated code is the same with your input number'
					
					print 'The system stops the code generation process'
					print 'Thank you'
				

			else:

				# using the string "-----"
				storedStr = ''
				stoppingStr = '-----'

				while True:
					x = np.reshape(pattern, (1, len(pattern), 1))
					x = x / float(numOfVocab)
					prediction = textGenModel.predict(x, verbose=0)
					index = np.argmax(prediction)
					
					result = intToChar[index]
					
					storedStr = storedStr + result

					# Checking whether there is the string "-----" in the stored string
					if stoppingStr in storedStr:
						break

					seq_in = [intToChar[value] for value in pattern]
					
					# PRINT AND APPEND
					print result
					generatedSequences.append(result)

					pattern.append(index)
					pattern = pattern[1:len(pattern)]


			
			print generatedSequences
			print '[INFO] Done\n'


			# Combining the generated code from the Encoder-Decoder model and the Text Generator model
			print '[INFO] Combining the generated code from the Encoder-Decoder and the Text Generator model'
			
			generatedSequences.insert(0, sequences[0])
			
			print '[INFO] Done\n'


			print '[INFO] Saving the result in a file...'
			
			np.savetxt('test_result', generatedSequences, newline='', fmt='%s')
			
			print '[INFO] Done\n'


			# PROMPT THE USER WHETHER TO PROCEED TO THE NEXT TESTING SAMPLE OR NOT
			print '\n'
			print 'The processing of this testing sample has completed'
			print ('+ {}'.format(td))
			print '\n'

			proceedToNextTestingSample = int(raw_input('Proceed to the next sample? [0/1]'))

			if proceedToNextTestingSample == 0:
				break
