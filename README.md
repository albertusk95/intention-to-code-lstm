# Source Code Generation Based On User Intention Using LSTM Networks

## Identification

### Research Topic

Source Code Generation Based On User Intention Using LSTM Networks

- _The World of Automatic Programming_ -

### Current Problems

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
  <li><b>Problem Statement (natural language)</b><br/>
    <p>There are various ways to state a problem query in natural language which means they are represented in casual and imprecise specifications. Therefore, it would not be effective if the natural language training data contains all the possible statements. In addition, the number of samples in the training data was also limited and there was a possibility that the new input representation could not be addressed by the trained model since its similarity was not close to the representation in the training data.</p>
    <p>Therefore, the natural language samples were represented as structured text having template in this form: action object type output. The action part denoted what the user wants to do, such as ‘find’, ‘check’, ‘confirm’, etc. The object part denoted the target of the action, such as ‘min number’, ‘sum of subset’, and so on. The type part denoted the data structure in which the action should be applied to, such as ‘array’, ‘list’, and so on. The output part denoted what should be shown to the user as the result, such as ‘yes no’, ‘number’, and so on. Fig. III.1. depicts the structured format of the training data of the 1 st problem, while Fig. III.2. depicts the structured format of the training data of the 2 nd problem. Based on the used representation of the natural language, this project used these following structured format for every problem type:
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
  
  <li><b>Source Code</b><br/>
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

**B. Experiments**



### Results
