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

**B. Experiments**

  <p align="center">
    <img src ="https://github.com/albertusk95/intention-to-code-lstm/blob/master/assets/img/two_experiments.png?raw=true"/>
  </p>
  <p align="center">
    Fig. 5. Two experiments conducted in the research
  </p>
  <br/>
  <p>The performance of several variants of LSTM networks, such as the Encoder-Decoder, Normal Sequential, and the combination of both of the models was examined via two experiments. Based on Fig. 5., the first experiment examined the performance of the combination of the Encoder-Decoder and the Normal Sequential model, whereas the second experiment examined the performance of the Normal Sequential model only.</p>
  <p>The Encoder-Decoder model captures the semantic representation of input whereas the Normal Sequential model predicts the next character given some previous characters as the seed. Based on these concepts, the first experiment tried to generate the final source code in which the decoded semantic representation became the seed for the Normal Sequential model, whereas the second experiment tried to generate the final source code in which some characters of a training sample became the seed.</p>

**B.1 Problem Type Classification**

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

**B.2 Experiment 1**

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
  
**B.3 Experiment 2**

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
  <br/>
  
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
  <b>Document similarity section:</b>

  Testing sample A

  <p align="center">
    <img src ="https://github.com/albertusk95/intention-to-code-lstm/blob/master/assets/img/test_ed_sim_2A.png?raw=true"/>
  </p>
  <p align="center">
    Fig. 14. Computing Similarity Score
  </p>
  <br/>

  Testing sample B

  <p align="center">
    <img src ="https://github.com/albertusk95/intention-to-code-lstm/blob/master/assets/img/test_ed_sim_2B.png?raw=true"/>
  </p>
  <p align="center">
    Fig. 15. Computing Similarity Score
  </p>
  
  </p>
  
  <br/>

  <p>
  <b>Generated code by the Encoder-Decoder model (seed code):</b>

  Testing sample A

  <p align="center">
    <img src ="https://github.com/albertusk95/intention-to-code-lstm/blob/master/assets/img/test_ed_seed_2A.png?raw=true"/>
  </p>
  <p align="center">
    Fig. 16. Seed code generated by the Encoder-Decoder
  </p>
  <br/>

  Testing sample B

  <p align="center">
    <img src ="https://github.com/albertusk95/intention-to-code-lstm/blob/master/assets/img/test_ed_seed_2B.png?raw=true"/>
  </p>
  <p align="center">
    Fig. 17. Seed code generated by the Encoder-Decoder
  </p>
  
  </p>
  
  <br/>

  <p>
  <b>Final source code (stored in a file):</b>

  Testing sample A

  <p align="center">
    <img src ="https://github.com/albertusk95/intention-to-code-lstm/blob/master/assets/img/test_ed_fin_2A.png?raw=true"/>
  </p>
  <p align="center">
    Fig. 18. Final source code
  </p>
  <br/>
  
  Testing sample B

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
  User input:
  A. find the position of uncle johny after sorting
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
  User input:
  A. compute the minimum diff between two horses
  B. find out the number of characters of x which is also resides in y
  </b>
  
  <p>
  <b>Document similarity section:</b>
  
  Testing sample A
  
  <p align="center">
    <img src ="https://github.com/albertusk95/intention-to-code-lstm/blob/master/assets/img/test_exp02_sim_2A.png?raw=true"/>
  </p>
  <p align="center">
    Fig. 23. Computing similarity score
  </p>
  <br/>
  
  Testing sample B
  
  <p align="center">
    <img src ="https://github.com/albertusk95/intention-to-code-lstm/blob/master/assets/img/test_exp02_sim_2B.png?raw=true"/>
  </p>
  <p align="center">
    Fig. 24. Computing similarity score
  </p>
  
  </p>
  
  <br/>
  
  <p>
  <b>Seed code:</b>
  
  Testing sample A
  
  <p align="center">
    <img src ="https://github.com/albertusk95/intention-to-code-lstm/blob/master/assets/img/test_exp02_seed_2A.png?raw=true"/>
  </p>
  <p align="center">
    Fig. 25. Seed code generated based on the random index
  </p>
  <br/>
  
  Testing sample B
  
  <p align="center">
    <img src ="https://github.com/albertusk95/intention-to-code-lstm/blob/master/assets/img/test_exp02_seed_2B.png?raw=true"/>
  </p>
  <p align="center">
    Fig. 26. Seed code generated based on the random index
  </p>
  
  </p>
  
  <br/>
  
  <p>
  <b>Final source code:</b>
  
  Testing sample A
  
  <p align="center">
    <img src ="https://github.com/albertusk95/intention-to-code-lstm/blob/master/assets/img/test_exp02_fin_2A.png?raw=true"/>
  </p>
  <p align="center">
    Fig. 27. Final source code
  </p>
  <br/>
  
  Testing sample B
  
  <p align="center">
    <img src ="https://github.com/albertusk95/intention-to-code-lstm/blob/master/assets/img/test_exp02_fin_2B.png?raw=true"/>
  </p>
  <p align="center">
    Fig. 28. Final source code
  </p>
  
  </p>
  <br/>
</p>
