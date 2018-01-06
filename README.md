# Source Code Generation Based On User Intention Using LSTM Networks

_The World of Automatic Programming_

<a href="https://github.com/albertusk95/intention-to-code-lstm">view on GitHub</a>

## Identification

### Research Topic

Source Code Generation Based On User Intention Using LSTM Networks

### Main Problems

<ul>
  <li>How to ensure that the user receives the correct source code representing the solution to the problem statement?</li>
  <li>How does the system work to understand the user intention and generate the desired code?</li>
</ul>

### Objectives

<ul>
  <li>Develop an automatic programming system that receives a problem statement written in natural language and then generates the relevant source code</li>
  <li>Examined the performance of several variants of LSTM networks, such as the Encoder-Decoder, Normal Sequential LSTM, and the combination of both of the models via two experiments</li>
</ul>

## Primary Works

### Methodology

**A. Training Samples**

<ul>
  <li>
    <b>Problem Statement (natural language)</b>
    <p>There are various ways to state a problem query in natural language which means they are represented in casual and imprecise specifications. Therefore, it would not be effective if the natural language training data contains all the possible statements. In addition, the number of samples in the training data was also limited and there was a possibility that the new input representation could not be addressed by the trained model since its similarity was not close to the representation in the training data.</p>
    <p>Therefore, the natural language samples were represented as structured text having template in this form: action object type output. The action part denoted what the user wants to do, such as ‘find’, ‘check’, ‘confirm’, etc. The object part denoted the target of the action, such as ‘min number’, ‘sum of subset’, and so on. The type part denoted the data structure in which the action should be applied to, such as ‘array’, ‘list’, and so on. The output part denoted what should be shown to the user as the result, such as ‘yes no’, ‘number’, and so on. Fig. 1. depicts the structured format of the training data of the 1 st problem, while Fig. 2. depicts the structured format of the training data of the 2 nd problem. Based on the used representation of the natural language, this project used these following structured format for every problem type:
    <ol>
      <li><b>Problem 1</b>: find string position array index</li>
      <li><b>Problem 2</b>: calculate minimum diff array difference</li>
      <li><b>Problem 3</b>: find same chars two str total</li>
    </ol>
    </p>
  <br/>
  <p align="center">
    <img src ="https://github.com/albertusk95/intention-to-code-lstm/blob/master/assets/img/nl_samples_1.png?raw=true"/>
  </p>
  <p align="center">
    Fig.1. Structured format of the training data of the 1st problem
  </p>
  <br/>
  <p align="center">
    <img src ="https://github.com/albertusk95/intention-to-code-lstm/blob/master/assets/img/nl_samples_2.png?raw=true"/>
  </p>
  <p align="center">
    Fig.2. Structured format of the training data of the 2nd problem
  </p>
  </li>
  
  <li><b>Source Code</b>
    <p>
      The selected training data were in their original format, which means all the elements in the data, such as space, lower case, upper-case, and indentation were preserved. Each problem type was collected in the separate files which means there were 3 different files containing the corresponsing problem type. The total number of samples for each problem type was 120 in which in this case the total number of training samples were 360.
    </p>
    <p>
      In every file, each sample was separated by special sequence of characters, namely ‘-----’ (five dashes characters). This separator was used to fetch each sample and insert it into a list of training sample later.
    </p>
    <p>
      However, this kind of original format was only applied for character level modelling in which the network model received a sequence of characters (one-by-one) in each timestep to predict the next one. Since there was a need to use the word level modelling in the Encoder-Decoder model (the first experiment), the format of the training data was modified because in this case the network model took an input in the form of word instead of character. The training data was modified manually.
    </p>
    <p>
      To get a better understanding, Fig. 3. and Fig. 4. depict the training samples from the first problem’s type in the original and modified format respectively. In addition, this is an example behind the reason of modifying the training data. Suppose there are 3 samples in the original format, namely print(‘abc’), print(), and print(1). If only these samples format are used, the network model will presume that they are different words and it makes the size of vocabulary become larger since now our vocabulary will contain those 3 words. Moreover, those 3 words are not appropriate enough to become the vocabulary since the possibility for them to exist in the training data with large amount is small (there would be many representation on how to show an output with print command). Even though the exact amount of vocabulary is specified, our vocabulary would contain only the insignificant words with small frequency. Therefore, some modifications are needed in this case, for example separating a sequence of characters into some
appropriate words that can be included In the vocabulary and providing a single space at the end of every line in the training data. After the modification, they would be like this at the end: print ( ‘abc’ ), print ( ), and print ( 1 ). In this case, the network model will recognize print, (, ), abc, and 1 as a part of the vocabulary. As the result, the number of vocabulary increases but it is not a problem since the total number of valid words having large frequency in the training data can be specified, which means the vocabulary will only contain significant words, such as print, (, and ). These significant words are common in Python programming language which means the possibility for them to exist in the training data with large amount is big. In this case, the vocabulary would contain only the significant words which are sorted based on their frequency in the training data (from big to small frequency). For the sake of clarity, this is an example on how the system builds the vocabulary. Suppose the unique words contained in our training data are list, print, for, while, int, range, x, y, and myVarName. Also, the total number of occurrence (frequency) for each word are 30, 25, 29, 30, 20, 15, 2, 3, and 1. After being sorted from the biggest one to the smallest one, we got 30, 30, 29, 25, 20, 15, 3, 2, and 1. If there is a specification stating that the vocabulary only contains 6 words with the biggest frequency, the vocabulary will contain list, while, for, print, int, and range. For the remaining words (x, y, and myVarName), they will be changed with a special token called ‘UNK’ (unknown).
    </p>
    <br/>
    <p align="center">
      <img src ="https://github.com/albertusk95/intention-to-code-lstm/blob/master/assets/img/original_code_samples.png?raw=true"/>
    </p>
    <p align="center">
      Fig.3. Code samples in the original format
    </p>
    <br/>
    <p align="center">
      <img src ="https://github.com/albertusk95/intention-to-code-lstm/blob/master/assets/img/word_code_samples.png?raw=true"/>
    </p>
    <p align="center">
      Fig.4. Code samples in the modified format
    </p>
  </li>
</ul>

-----

**B. Model Design**

**B.1 Encoder-Decoder LSTM**

<p>
  Long Short Term Memory (LSTM) is a special type of Recurrent Neural Networks (RNNs). It has a capability to learn the pattern of time series input or in other words it can learn the long term dependencies. One application of LSTM is to build a text generator machine that receives few characters and then predicts the next character one by one in each timestep. To build this kind of machine we can use vanilla LSTM which only uses several layers with specified number of hidden units. In this case, the length of input sample is the same with the length of output sample. Also, the length of all the input samples are the same. The problem comes when the length of input sample is not the same as the length of output sample. Moreover, the condition when the length of all the input samples are not the same must be addressed as well.
</p>

<p>
  Based on this project in which the length of natural language samples were not the same with the length of source code samples, this problem was relevant. Therefore, the Encoder-Decoder LSTM was applied for the 1 st experiment. To accomplish this task, the Keras framework was used to build an abstraction for the neural network, including the network model creation, weight loading and saving, model training (fit), and model testing (predict).
</p>

<p>
  Basically, this model had two LSTM networks, the first one acted as an encoder and the second one acted as a decoder. The encoder received the sequence of words from the natural language samples and as the result it gave a vector capturing the semantic meaning within the input sequences. Then, this semantic vector was passed to the decoder in which it would be analyzed and based on the semantic contained in the vector, the decoder generated the corresponding output sequence.
</p>

<p>
These were the specifications of the Encoder-Decoder model used in the 1st experiment:
<ul>
  <li>
    Using an Embedding layer as the first layer of the network.
    <p>
    Based on the Keras documentation, this layer turns positive integers (indexes) into dense vectors of fixed size, for example [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]. This layer can only be used as the first layer in a model. The model took a vector with size (batchSize, inputLength) as the input, in which the inputLength was the maximum length of natural language samples.
    </p>
    <p>
    In addition, there were several important parameters for this layer, such as vocabulary size, output dimension, and input length. The vocabulary size was used to validate whether the index number in the input sequence exceed the value of vocabulary size. The output dimension denoted the embedding size. The research used 1000 as the embedding size. Finally, the output of this layer had (batchSize, inputLength, outputDimension) as the dimension.
    </p>
  </li>
  <li>
    <p>
    For the Encoder network, one LSTM layer with 1000 units was used. The used LSTM was not stateful which means that the final state of a sample in a batch was not used as the initial state of a sample in the next batch. For example, the final state of the 1st sample in the 1 st batch is not used as the initial state of the 1 st sample in the 2 nd batch.
    </p>
  </li>
  <li>
    After the encoder network, there was a RepeatVector network.
    <p>
    To understand how this network works, suppose the output shape given by the encoder is (None, 32). In this case, ‘None’ is the batch dimension. If a RepeatVector network with a number 3 as the parameter is added, the output shape becomes (None, 3, 32). So basically, this network builds the same vector as much as the specified number. In other words, if the encoder gives [1, 2, 3] as the output vector and there is a need to build 3 other vectors, the method RepeatVector(3) can be used and [[1, 2, 3], [1, 2, 3], [1, 2, 3]] becomes the result.
    </p>
    <p>
      However, there might be a question regarding the reason of using this RepeatVector network. The basic idea was to make the Decoder network to act in the same wayas the vanilla LSTM, in which it receives the input sequence having the same length with the output sequence. To make it becomes more clear, the output vector given by the Encoder network was repeated n times, with n was the length of output sequence or in this case it was the source code samples. Since the length of output sequence was different, there was a need to specify an exact amount of characters so that the RepeatVector works. The simplest way to do this was by computing the length of every source code sample and then find the longest one. Afterwards, the Decoder network acted in the same way as the vanilla LSTM.
    </p>
  </li>
  <li>
    <p>
    For the Decoder network, the specifications were by using 4 LSTM layers with 1000 units in each layer. Moreover, the Decoder network returned sequences of words rather than single values. The sequence of words represented the generated source code. In addition, the return value was set to be sequences because the next layer (TimeDistributed wrapper layer) needed input in 3 dimension.
    </p>
  </li>
  <li>
    <p>
    Afterwards, a TimeDistributed layer wrapping a Dense layer was added to the network so that each unit or neuron was connected to each neuron in the next layer. The application of TimeDistributed layer made it possible to use one layer (the Dense layer) to every element in the sequence independently. That layer has the same weights for every element and returns the sequence of words processed independently.
    </p>
    <p>
    For the sake of clarity, suppose the input sequence has n words and an Embedding layer with embSize as the dimensions is used which means that our input sequence dimension is (n, embSize). Then, the sequence is fed to the LSTM having lstmDim as the output dimension and the output sequence with (n, lstmDim) as the dimension is retrieved. Suppose a TimeDistributed layer to wrap a Dense layer is used and has denseDim as the output dimension. By doing so, the wrapper will apply the Dense layer to each of the output vector (n times). The effect of this application is that the final output will have (n, denseDim) as the dimension.  
    </p>
  </li>
  <li>
    <p>
    The last element of this network model was the softmax function acting as the activation function. This function produced an array with dimension (batchSize, numOfOutputSeq, numOfVocab), in which batchSize was the number of batch used to update the weights, numOfOutputSeq was the maximum length of the source code samples which was explained in the 3rd point about the RepeatVector, and numOfVocab denoted the amount of words in the vocabulary. Basically, for each element in the output sequence, this softmax function gave the probability of every word in the vocabulary to be the most suitable output.  
    </p>
  </li>
</ul>
</p>

**B.2 Normal Sequential LSTM**

<p>
This model had simpler working concept than the Encoder-Decoder model. It received 90 characters of source code generated by the Encoder-Decoder model and then generated the final source code. This model used different training samples compared with the Encoder-Decoder model in which it learnt the mapping pattern between the characters in the raw text of source code training data and did not use the raw text of natural language training data.
</p>

<p>
  These were the specifications of this model:
  <ul>
    <li>
      <p>
      The network used 2 LSTM layers with 256 units in each layer
      </p>
    </li>
    <li>
      After each LSTM layer, a Dropout layer with probability 0.2 was added. The probability 0.2 means that one in five inputs will be randomly excluded from each cycle of updating process.
    <p>
    Dropout is a regularization technique to prevent overfitting. The basic concept is this technique randomly select neurons to be ignored during training. When a neuron is dropped out, it can not contribute to the network and receive some values in the backpropagation process. Moreover, as the result of the dropout, other neurons will take the responsibility of the dropped out neuron to make predictions in the training process. One of the effect is there would be multiple independent internal representations being learnt by the model.
      </p>
      <p>
      Furthermore, the model will become less sensitive to the spesific weights of neurons. This causes the model to make better generalization towards the new data and prevent the overfitting problem.  
      </p>
    </li>
    <li>
      <p>
      The next layer was Dense layer with the total number of words in the vocabulary as the dimension.
      </p>
    </li>
    <li>
      <p>
      The last layer used softmax as the activation function. The output of this function is modelled as the probability distribution over K different possible outcomes. Specifically, it shows the probability of each word in the vocabulary to be the most suitable output for the spesific element in the output vector. The probabilities add up to 1.
      </p>
    </li>
  </ul>
</p>

-----

**C. Model Training**

**C.1 Encoder-Decoder LSTM**

<p>
These were the specifications of the Encoder-Decoder model training:
</p>

<ul>
  <li>
    <p>
    The first thing to do was building the network model as well as its initial weights. The initial weights were assigned randomly by the Keras framework which means that this initial network was still empty, had big loss value and small accuracy. The created model applied the design which I’ve explained in the ‘Model Design’ section.
    </p>
  </li>
  <li>
    <p>
    The created network model was compiled before being returned to start the training. When a network model was compiled, several parameters were provided, such as the loss function, optimizer, and metrics. Loss function measures the difference between the predicted output and the desired output. There are several loss function that can be used, such as mean-squared error, hinge loss, logistic loss, cross entropy loss, etc. For this research, the categorical cross entropy was used as the loss function. It was a suitable choice since the model was trained to provide multiclass prediction. Moreover, this loss function gave better result than squared mean error because it makes the weight updating process become faster since there is no any derivative elements which might give a value approaching zero which denotes the difference between the current and the updated weight.
    </p>
    <p>
    Optimizer was used to increase the speed of the decrement of loss value. Suppose we have a graph describing a loss function. The initial loss value resides in a certain point. The objective is how to make the loss value achieves the global optima (the minimum point) as fast as possible. One of the most common used optimizer is the gradient descent optimization. It has several algorithms, such as ADAM, RMSProp, Adagrad, Adadelta, Nadam, Momentum, etc. The ADAM optimizer was used in this research. Metrics stated the type of measurement for judging the performance of trained model. It is similar with the loss function, except that the value provided by the metrics is not used for the training process. In this research, the accuracy metric was used to compute the accuracy score of the model. 
    </p>
  </li>
  <li>
    <p>
    Checking whether there was any saved model or not. In addition, the system could save the weights after the specified epoch and then loaded them back when the new training was started. As the result, if there was any saved model, the system would use that model as the initial state, whereas the system would use the empty model created at the first step as the initial state if there is no any saved model.
    </p>
  </li>
  <li>
    Creating one-hot vector for the source code training data. This step needed several parameters, such as the list of zero-padded source code index, maximum length of source code samples, and the source code’s dictionary for mapping the word to index.
    <p>
    Basically, one-hot vector is a type of vector having a single value of 1 as its element, whereas the rest are 0. For example, if the vocabulary contains these following words: ZERO print array if while UNK, and if our input sequence is print ‘ok’ if success, then the one-hot vector of this input sequence is [ [0 1 0 0 0 0], [0 0 0 0 0 1], [0 0 0 1 0 0], [0 0 0 0 0 1] ].
    </p>
    <p>
    One-hot vector is only one example of word embedding techniques and it is necessary to convert the word string into its numeric representation since the neural networks only work with number.
    </p>
  </li>
  <li>
    <p>
      Determining the location of file for saving the model. In addition, a checkpoint in which the system automatically saved the model after each epoch was created. The system only saved the best model based on the loss value. The model in an epoch was considered as a better model than the one in previous epoch if its loss value was smaller than the previous one.
    </p>
  </li>
  <li>
    <p>
      Calling the fit method to start the training. This method required several parameters, such as list of zero-padded natural language training data, list of source code training data which was represented in their one-hot vector, number of epochs, batch size, and list of callbacks (checkpoint). This research used 100 epochs with batch size 10 which means that the system updated the weight after being trained with 10 samples. After all the samples were trained, the same process was repeated again for 100 times.
    </p>
  </li>
</ul>

**C.2 Normal Sequential LSTM**

<p>
  The applied specifications were the same as ones used in the model training of the Encoder-Decoder model. The difference is only in the 6 th step in which the batch size of this model training was 128.
</p>

<p>
  The Normal Sequential model only used one type of training data, namely the source code samples rather than two types of training data (natural language and source code) which were used by the Encoder-Decoder model. These were the specifications of the Normal Sequential model training:
</p>
  
<ul>
  <li>
    <p>
      Loading the raw text of source code training data
    </p>
  </li>
  <li>
    <p>
      Converting the raw text of source code training data into lower case
    </p>
  </li>
  <li>
    <p>
      Building a vocabulary based on the unique characters in the raw text of source code training data
    </p>
  </li>
  <li>
    <p>
      Creating a dictionary for mapping the unique characters in the raw text of source code training data to integers
    </p>
  </li>
  <li>
    <p>
      Computing the length of vocabulary and characters in the raw text of source code training data
    </p>
  </li>
  <li>
    Building the actual training data.
    <p>
      Basically, the system took few characters from the raw text of source code training data as the input sample, and then took the next one character as the output sample. For example, suppose the raw text is for x in range(k): print x. The input sample having 5 characters will be [ ‘for x’ → ‘ ’ ], [ ‘or x ’ → ‘i’ ], [ ‘r x i’ → ‘n’ ], [ ‘ x in’ → ‘ ’ ], [ ‘x in ’ → ‘r’ ], [ ‘ in r’ → ‘a’ ], and so on.
    </p>
    <p>
    Afterwards, each character in the training samples were converted into integer since neural networks only work with numbers. As the result, there were two lists in which the first one contained the list of integers of input samples and the second one contained the list of integer of output sample.
    </p>
  </li>
  <li>
    <p>
      Reshaping the input samples so that the dimension became [numberOfSamples, lengthOfSequence, features]. The numberOfSamples denoted the amount of input samples. The lengthOfSequence denoted the length of one input sample in which this research used 100 characters. The features denoted the way to read the elements of an array using the specified index order.
    </p>
  </li>
  <li>
    <p>
      Normalizing the input samples. Basically, normalization is a technique to give weights to a term. In this research, the term was the index of input samples and the weight was the amount of vocabulary. The elements were normalized by dividing each element with the weight.
    </p>
  </li>
  <li>
    <p>
      Creating one-hot vector for the output samples. The detail explanations are provided in section C.1 about the model training for the Encoder-Decoder model
    </p>
  </li>
  <li>
    <p>
      Building the network model of the Normal Sequential LSTM
    </p>
  </li>
</ul>

-----

**D. Experiments**

  <p align="center">
    <img src ="https://github.com/albertusk95/intention-to-code-lstm/blob/master/assets/img/two_experiments.png?raw=true"/>
  </p>
  <p align="center">
    Fig. 5. Two experiments conducted in the research
  </p>
  <br/>
  <p>The performance of several variants of LSTM networks, such as the Encoder-Decoder, Normal Sequential, and the combination of both of the models was examined via two experiments. Based on Fig. 5., the first experiment examined the performance of the combination of the Encoder-Decoder and the Normal Sequential model, whereas the second experiment examined the performance of the Normal Sequential model only.</p>
  <p>The Encoder-Decoder model captures the semantic representation of input whereas the Normal Sequential model predicts the next character given some previous characters as the seed. Based on these concepts, the first experiment tried to generate the final source code in which the decoded semantic representation became the seed for the Normal Sequential model, whereas the second experiment tried to generate the final source code in which some characters of a training sample became the seed.</p>

**D.1 Problem Type Classification**

  <p align="center">
    <img src ="https://github.com/albertusk95/intention-to-code-lstm/blob/master/assets/img/base_prob_stat.png?raw=true"/>
  </p>
  <p align="center">
    Fig. 6. Base Problem Statement
  </p>
  <br/>
  <p align="center">
    <img src ="https://github.com/albertusk95/intention-to-code-lstm/blob/master/assets/img/structured_format.png?raw=true"/>
  </p>
  <p align="center">
    Fig. 7. Structured Format
  </p>
  <br/>

**D.2 Experiment 1**

  <p align="center">
    <img src ="https://github.com/albertusk95/intention-to-code-lstm/blob/master/assets/img/experiment_1_flow.png?raw=true"/>
  </p>
  <p align="center">
    Fig. 8. General Working Process of the 1st Experiment
  </p>
  <br/>
  <p>
  The general working process of the 1st experiment is shown in the Fig. 8.. In this experiment, the user gave a problem statement in natural language to the system. Then, the system compared the problem statement with each of the base problem statement which is shown in Fig. 6.. Based on the most similar base problem statement, the system retrieved the representation of user input in structured format. Fig.7. depicts the list of problem statements in their structured format.
  </p>
  <p>
  Since the neural networks only deal with numbers, the next stage was preprocessing the structured text in which the primary goal of the stage was to convert the text representation into numerical format. There were several steps to accomplish the goal, such as reversing the order of words, applying zero padding, converting word into its corresponding index in the vocabulary, and so on. After being preprocessed, each word in the structured text was converted into its index which was in numerical format.
  </p>
  <p>
  Afterwards, the list of index from the preprocessing step became the input for the Encoder-Decoder model. Starting from the Encoder model, it received the list of index as the input sequence and captured the semantic of the sequence. Basically, the output of the Encoder model was an encoded representation of the input sequence which could be seen as the semantic of the user input.
  </p>
  <p>
  Based on the encoded representation, the Decoder model generated some words as the output sequence which was used as the seed. The seed code was not the final source code, but it was used as the initial sequence of words for the Normal Sequential model. By using the seed code, the Normal Sequential model generated the final source code.
  </p>
  <br/>
  
**D.3 Experiment 2**

  <p align="center">
    <img src ="https://github.com/albertusk95/intention-to-code-lstm/blob/master/assets/img/experiment_2_flow.png?raw=true"/>
  </p>
  <p align="center">
    Fig. 9. General Working Process of the 2nd Experiment
  </p>
  <br/>
  <p>
  The general working process of the 2 nd experiment is shown in the Fig. 9.. In this experiment, the user gave a problem statement in natural language to the system. Then, the system compared the problem statement with each of the base problem statement (shown in Fig. 6.). Based on the most similar base problem statement, the system retrieved the representation of user input in structured format (shown in Fig. 7.).
  </p>
  <p>
  Afterwards, the system searched for a random index starting from the first to last index of the corresponding problem type in the training samples. The list of index for each problem is shown in Fig. 10. The problem type was determined from the selected base problem statement in the previous step.
  </p>
  <br/>
  <p align="center">
    <img src ="https://github.com/albertusk95/intention-to-code-lstm/blob/master/assets/img/listofprobidx.png?raw=true"/>
  </p>
  <p align="center">
    Fig. 10. List of minimum and maximum index for each problem
  </p>
  <br/>
  <p>
  Based on the chosen random index, the system checked whether the length of training sample having that index was less than the length of seed code. If the condition was true, the system added n times of empty space characters in front of the training sample so that it had the same length as the seed code. On the other hand, the system took the first m characters of training sample in which m was the length of seed code. By using the seed code, the Normal Sequential model generated the final source code.
  </p>
  
### Results

**Experiment 1 (the Encoder-Decoder and the Normal Sequential Model)**

<p>
  <b>
  User input:<br/>
  A. find the position of uncle johny after sorting<br/>
  </b>
  
  <p>
  <b>Document similarity section:</b>

  <p align="center">
    <img src ="https://github.com/albertusk95/intention-to-code-lstm/blob/master/assets/img/test_ed_sim_1.png?raw=true"/>
  </p>
  <p align="center">
    Fig. 11. Computing similarity score
  </p>
  
  </p>
  
  <br/>
  
  <p>
  <b>Generated code by the Encoder-Decoder model (seed code):</b>
  
  <p align="center">
    <img src ="https://github.com/albertusk95/intention-to-code-lstm/blob/master/assets/img/test_ed_seed_1.png?raw=true"/>
  </p>
  <p align="center">
    Fig. 12. Seed code generated by the Encoder-Decoder
  </p>
  
  </p>
  
  <br/>
  
  <p>
  <b>Final source code (stored in a file):</b>
  
  <p align="center">
    <img src ="https://github.com/albertusk95/intention-to-code-lstm/blob/master/assets/img/test_ed_fin_1.png?raw=true"/>
  </p>
  <p align="center">
    Fig. 13. Final source code
  </p>
  
  </p>
  
  <br/>
</p>

---

<p>
  <b>
  User input:<br/>
  A. compute the minimum diff between two horses<br/>
  B. find out the number of characters of x which is also resides in y<br/>
  </b>

  <p>
  <b>Document similarity section:</b><br/>

  Testing sample A

  <p align="center">
    <img src ="https://github.com/albertusk95/intention-to-code-lstm/blob/master/assets/img/test_ed_sim_2A.png?raw=true"/>
  </p>
  <p align="center">
    Fig. 14. Computing Similarity Score
  </p>
  
  <br/>

  Testing sample B<br/>

  <p align="center">
    <img src ="https://github.com/albertusk95/intention-to-code-lstm/blob/master/assets/img/test_ed_sim_2B.png?raw=true"/>
  </p>
  <p align="center">
    Fig. 15. Computing Similarity Score
  </p>
  
  </p>
  
  <br/>

  <p>
  <b>Generated code by the Encoder-Decoder model (seed code):</b><br/>

  Testing sample A

  <p align="center">
    <img src ="https://github.com/albertusk95/intention-to-code-lstm/blob/master/assets/img/test_ed_seed_2A.png?raw=true"/>
  </p>
  <p align="center">
    Fig. 16. Seed code generated by the Encoder-Decoder
  </p>
  <br/>

  Testing sample B<br/>

  <p align="center">
    <img src ="https://github.com/albertusk95/intention-to-code-lstm/blob/master/assets/img/test_ed_seed_2B.png?raw=true"/>
  </p>
  <p align="center">
    Fig. 17. Seed code generated by the Encoder-Decoder
  </p>
  
  </p>
  
  <br/>

  <p>
  <b>Final source code (stored in a file):</b><br/>

  Testing sample A

  <p align="center">
    <img src ="https://github.com/albertusk95/intention-to-code-lstm/blob/master/assets/img/test_ed_fin_2A.png?raw=true"/>
  </p>
  <p align="center">
    Fig. 18. Final source code
  </p>
  <br/>
  
  Testing sample B<br/>

  <p align="center">
    <img src ="https://github.com/albertusk95/intention-to-code-lstm/blob/master/assets/img/test_ed_fin_2B.png?raw=true"/>
  </p>
  <p align="center">
    Fig. 19. Final source code
  </p>
  
  </p>
  
  <br/>
</p>

<br/>

**Experiment 2 (the Normal Sequential Model)**

<p>
  <b>
  User input:<br/>
  A. find the position of uncle johny after sorting<br/>
  </b>
  
  <p>
  <b>Document similarity section:</b>

  <p align="center">
    <img src ="https://github.com/albertusk95/intention-to-code-lstm/blob/master/assets/img/test_exp02_sim_1.png?raw=true"/>
  </p>
  <p align="center">
    Fig. 20. Computing similarity score
  </p>
  
  </p>
  
  <br/>

  <p>
  <b>Seed code:</b>
  
  <p align="center">
    <img src ="https://github.com/albertusk95/intention-to-code-lstm/blob/master/assets/img/test_exp02_seed_1.png?raw=true"/>
  </p>
  <p align="center">
    Fig. 21. Seed code generated based on the random index
  </p>
  
  </p>
  
  <br/>
  
  <p>
  <b>Final source code (stored in a file):</b>
  
  <p align="center">
    <img src ="https://github.com/albertusk95/intention-to-code-lstm/blob/master/assets/img/test_exp02_fin_1.png?raw=true"/>
  </p>
  <p align="center">
    Fig. 22. Final source code
  </p>
  
  </p>
  
  <br/>
</p>

---

<p>
  <b>
  User input:<br/>
  A. compute the minimum diff between two horses<br/>
  B. find out the number of characters of x which is also resides in y<br/>
  </b>
  
  <p>
  <b>Document similarity section:</b><br/>
  
  Testing sample A
  
  <p align="center">
    <img src ="https://github.com/albertusk95/intention-to-code-lstm/blob/master/assets/img/test_exp02_sim_2A.png?raw=true"/>
  </p>
  <p align="center">
    Fig. 23. Computing similarity score
  </p>
  
  <br/>
  
  Testing sample B<br/>
  
  <p align="center">
    <img src ="https://github.com/albertusk95/intention-to-code-lstm/blob/master/assets/img/test_exp02_sim_2B.png?raw=true"/>
  </p>
  <p align="center">
    Fig. 24. Computing similarity score
  </p>
  
  </p>
  
  <br/>
  
  <p>
  <b>Seed code:</b><br/>
  
  Testing sample A
  
  <p align="center">
    <img src ="https://github.com/albertusk95/intention-to-code-lstm/blob/master/assets/img/test_exp02_seed_2A.png?raw=true"/>
  </p>
  <p align="center">
    Fig. 25. Seed code generated based on the random index
  </p>
  <br/>
  
  Testing sample B<br/>
  
  <p align="center">
    <img src ="https://github.com/albertusk95/intention-to-code-lstm/blob/master/assets/img/test_exp02_seed_2B.png?raw=true"/>
  </p>
  <p align="center">
    Fig. 26. Seed code generated based on the random index
  </p>
  
  </p>
  
  <br/>
  
  <p>
  <b>Final source code:</b><br/>
  
  Testing sample A
  
  <p align="center">
    <img src ="https://github.com/albertusk95/intention-to-code-lstm/blob/master/assets/img/test_exp02_fin_2A.png?raw=true"/>
  </p>
  <p align="center">
    Fig. 27. Final source code
  </p>
  <br/>
  
  Testing sample B<br/>
  
  <p align="center">
    <img src ="https://github.com/albertusk95/intention-to-code-lstm/blob/master/assets/img/test_exp02_fin_2B.png?raw=true"/>
  </p>
  <p align="center">
    Fig. 28. Final source code
  </p>
  
  </p>
  <br/>
</p>

## Conclusions

<p>
  From the research that has been done about the source code generation based on user intention using LSTM networks, the author concludes that:
</p>

<ul>
  <li>
    <p>
      The author has developed an automatic programming system that generates source code based on the natural language input using LSTM networks
    </p>
  </li>
  <li>
    <p>
    The performance of several variants of LSTM networks, such as the Encoder-Decoder and Normal Sequential model was examined. The Encoder-Decoder model gave 1.4060 as the final loss value and 61.55% as the final accuracy score. On the other hand, the Normal Sequential model gave 0.2996 as the final loss value and 90,56% as the final accuracy score. Based on the result, the sequence of words in the 2 nd experiment was more coherence than the ones generated in the 1 st experiment. It was caused by the fact that the seed code in the 2 nd experiment was taken directly from the training data, which means that the sequence of characters in the seed code had already been in the structured and right format. On the other hand, the seed code generated by the Encoder-Decoder model was partially structured (does not always give the right sequence of characters)
    </p>
  </li>
</ul>

## References

[1] Bill Chambers. (December 21, 2014). _Basic Statistical NL Part 1 – Jaccard Similarity and TF-IDF_. Accessed on June 26, 2017, from http://billchambers.me/tutorials/2014/12/21/tf-idf-explained-in-python.html

[2] Bill Chambers. (December 22, 2014). _Basic Statistical NLP Part 2 – TF-IDF And Cosine Similarity_. Accessed on June 26, 2017, from http://billchambers.me/tutorials/2014/12/22/cosine-similarity-explained-in-python.html

[3] Christopher Olah. (August 27, 2015). _Understanding LSTM Networks_. Accessed on June 16, 2017, from http://colah.github.io/posts/2015-08-Understanding-LSTMs/

[4] Jahnavi Mahanta. (\_\__). _Keep it simple! How to understand Gradient Descent algorithm_. Accessed on June 19, 2017, from http://www.kdnuggets.com/2017/04/simple-understand-gradient-descent-algorithm.html

[5] Jason Brownlee. (June 20, 2016). _Dropout Regularization in Deep Learning Models With Keras_. Accessed on June 28, 2017, from http://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/

[6] Jason Brownlee. (May 17, 2017). _How to Use the TimeDistributed Layer for Long Short-Term Memory Networks in Python_. Accessed on June 24, 2017, from http://machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/

[7] [Mou et al.2015] Lili Mou, Rui Men, Ge Li, Li Zhang, and Zhi Jin. 2015. On end-to-end program generation from user intention by deep neural networks. _CoRR_, abs/1510.07211

[8] Nassim Ben. (March 13, 2017). _How to use return_sequences option and TimeDistributed layer in Keras_. Accessed on June 24, 2017, from https://stackoverflow.com/questions/42755820/how-to-use-return-sequences-option-and-timedistributed-layer-in-keras

[9] Raschka, Sebastian. 2015. _Python Machine Learning_. Packt Publishing: Birmingham

[10] Roopam Upadhyay. (April 3, 2016). _Intuitive Machine Learning : Gradient Descent Simplified_. Accessed on June 19, 2017, from http://ucanalytics.com/blogs/intuitive-machine-learning-gradient-descent-simplified/

[11] scikit-learn developers. \_\__. _sklearn.feature_extraction.text.TfidfVectorizer_. Accessed on June 26, 2017, from http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

[12] Sebastian Ruder. (June 15, 2017). _An overview of gradient descent optimization algorithms_. Accessed
on June 24, 2017, from http://ruder.io/optimizing-gradient-descent/

[13] The Scipy community. (_\__). _numpy.reshape_. Accessed on June 25, 2017, from 1.10.4/reference/generated/numpy.reshape.html https://docs.scipy.org/doc/numpy-

[14] Xi Victoria Lin, Chenglong Wang, Deric Pang, Kevin Vu, Luke Zettlemoyer, and Michael D. Ernst. Program synthesis from natural language using recurrent neural networks. Technical Report UW-CSE-17-03-01, University of Washington Department of Computer Science and Engineering, Seattle, WA, USA, March 2017

[15] _\__.(_\__). _Embedding_. Accessed on June 23, 2017, from https://keras.io/layers/embeddings/

[16] __\_. (June 22, 2017). _Softmax function_. Accessed on June 24, 2017, from https://en.wikipedia.org/wiki/Softmax_function

[17] _\__. (_\__). _Usage of loss functions_. Accessed on June 23, 2017, from https://keras.io/losses/

[18] _\__. (_\__). _Usage of metrics_. Accessed on June 23, 2017, from https://keras.io/metrics/

[19] _\__. (_\__). _Usage of optimizers_. Accessed on June 24, 2017, from https://keras.io/optimizers/

---

**Albertus Kelvin**<br/>
**Institut Teknologi Bandung**<br/>
**2018**<br/>
