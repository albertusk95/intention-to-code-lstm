
# File: master_exp02.py
# This file is the main controller of the 2nd experiment of this project

from master_exp02_nl_utils import *
from master_exp02_code_utils import *
from master_exp02_test import *
import numpy as np
import argparse


# Input parameters
param = argparse.ArgumentParser()
#param.add_argument('-ignlen', default=100)
param.add_argument('-seqlen', default=100)
params = vars(param.parse_args())

#IGN_LEN = params['ignlen']
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


# Constants
listOfProbMinIdx = [0, 120, 240]
listOfProbMaxIdx = [119, 239, 359]

# List of base problem statements
listOfBaseProbStatements = [
    							'find the position of "uncle johny" in the sorted playlist',
    							'report the minimum difference that is possible between 2 horses in the race',
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
print 'Welcome to the 2nd Experiment --- Main Program'
print '-------------------------------------------------------------------------'
print 'Topic: Source Code Generation Based On User Intention Using LSTM Networks'
print 'By: Albertus Kelvin'
print '-------------------------------------------------------------------------'
print '\n'

# Some Network Information
print 'Parameters information:'
print ('- number of sequences: {}'.format(SEQ_LEN))


# Get the number of generated code chars from the user
print 'Please read these rules:'
print '1. You can specify how the code generation process stops'
print '2. The possible choices are:'
print '   - specifying a number of generated characters'
print '   - using the string "-----"'
print '3. If you choose to use the string "-----", please give -1 as the input below\n'

numOfGeneratedCodeChars = int(raw_input('Number of generated code chars: '))



# Loading testing data
print '[INFO] Loading testing data...'

testingData = loadData('../../data/testing/testing_data')

print testingData
print '[INFO] Done\n'


# Fetching samples of testing data
print '[INFO] Splitting raw text of testing data to array of sentences...'

listOfTestingData = fetchNL(testingData)

print '[INFO] Done'


# Filtering the empty space in the list of NL
print '[INFO] Filtering out the empty space in the list of NL'

listOfTestingData = filter(lambda lotd: lotd != '', listOfTestingData)

print ('Total testing data: {}'.format(len(listOfTestingData)))

for lotd in listOfTestingData:
	print ('- {}'.format(lotd))

print'[INFO] Done\n'


# Initializing network model
print '[INFO] Loading training data for network model...'

codeTrainingDataOrigin = ''
for train_data_loc in listOfOriginCodeTrainingDataLocations:
	codeTrainingDataOrigin += loadData(train_data_loc)
	codeTrainingDataOrigin += '\n'

print codeTrainingDataOrigin

print '[INFO] Done\n'


print '[INFO] Converting training data to lower case...'

codeTrainingDataOrigin = convertToLowerCase(codeTrainingDataOrigin)

print '[INFO] Done\n'


print '[INFO] Splitting raw text of code to list of sentences...'

listOfCode = fetchCode(codeTrainingDataOrigin)

print '[INFO] Done\n'


print '[INFO] Creating mapping for char to int...'

charToInt = createMappingCharToInt(codeTrainingDataOrigin)

print charToInt
print '[INFO] Done\n'


print '[INFO] Creating mapping for int to char...'

intToChar = createMappingIntToChar(codeTrainingDataOrigin)

print intToChar
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

model = loadModel(reshapedAndNormalizedInput, vectorizedOutput)

print '[INFO] Done\n'


# Compute the similarity among the user input and the base problem statements
for td in listOfTestingData:
	
	idx, val = max(enumerate([computeCosineSimilarity(td, bp) for bp in listOfBaseProbStatements]), key=operator.itemgetter(1))
	print '[INFO] Computing similarity score...'
	print('Problem: {}'.format(td))
	print('Most similar with base number: {}'.format(idx))
	print('Base problem: {}'.format(listOfBaseProbStatements[idx]))
	print('Similarity score: {}'.format(val))
	print('Structured: {}'.format(listOfStructuredBaseProbStatements[idx]))
	print '[INFO] Done\n'


	# Getting random index of structured text in the training data to be fed into the text generator
	print '[INFO] Getting random index of structured text in the training data...'
	
	randIdxOfStructuredText = getRandIdxOfStructuredText(listOfProbMinIdx[idx], listOfProbMaxIdx[idx])
	
	print randIdxOfStructuredText
	print '[INFO] Done\n'


	# Creating the seed based on the random index of structured text
	print '[INFO] Creating the seed...'

	str_pattern_tmp = ''

	if len(listOfCode[randIdxOfStructuredText]) < SEQ_LEN:
		print 'Length of code sample is less than SEQ_LEN'
		delta = SEQ_LEN - len(listOfCode[randIdxOfStructuredText])
		str_pattern = ' '*delta + listOfCode[randIdxOfStructuredText]
		
	else:
		
		# Get the first SEQ_LEN characters
		str_pattern = listOfCode[randIdxOfStructuredText][:SEQ_LEN]
	
		'''
		if len(listOfCode[randIdxOfStructuredText]) < SEQ_LEN + IGN_LEN:
			print 'Length of code sample is less than SEQ_LEN + IGN_LEN'
			delta = SEQ_LEN + IGN_LEN - len(listOfCode[randIdxOfStructuredText])
			str_pattern_tmp = listOfCode[randIdxOfStructuredText] + ' '*delta
			str_pattern = str_pattern_tmp[IGN_LEN:]
		else:
			print 'Length of code sample is more than or equal with SEQ_LEN + IGN_LEN'
			str_pattern_tmp = listOfCode[randIdxOfStructuredText]
			str_pattern = listOfCode[randIdxOfStructuredText][IGN_LEN:IGN_LEN + SEQ_LEN]
		'''


	# TESTING
	#str_pattern = ' sort() [ = ] [ 1 [ - ] [ 0 ] for ( in ) ( 1 , ) ) 1 ) : [ ['

	print str_pattern
	print '[INFO] Done\n'


	# Converting pattern chars to index
	print '[INFO] Converting pattern chars to index...'
	
	pattern = [charToInt[idx] for idx in str_pattern]
	
	print pattern
	print '[INFO] Done\n'


	generatedSequences = []

			
	# Checking the number of generated code by the Encoder-Decoder model
	if numOfGeneratedCodeChars != -1:

		# using the specified number of generated chars
		numOfCurrentChars = len(listOfCode[randIdxOfStructuredText])

		if numOfCurrentChars < numOfGeneratedCodeChars:
			remainingNumOfGenChars = numOfGeneratedCodeChars - numOfCurrentChars
		
			for i in range(remainingNumOfGenChars):
				x = np.reshape(pattern, (1, len(pattern), 1))
				x = x / float(numOfVocab)
				prediction = model.predict(x, verbose=0)
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
			if numOfCurrentChars > numOfGeneratedCodeChars:
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
			prediction = model.predict(x, verbose=0)
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


	# Combining the seed code with the one generated by the Text Generator model
	print '[INFO] Combining the seed code with the one generated by the Text Generator model'
	
	generatedSequences.insert(0, str_pattern)
	
	'''
	if str_pattern_tmp == '':
		generatedSequences.insert(0, str_pattern)
	else:
		generatedSequences.insert(0, str_pattern_tmp[:IGN_LEN] + str_pattern)
	'''

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
