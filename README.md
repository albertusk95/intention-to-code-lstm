# Source Code Generation Based On User Intention Using LSTM Networks

## Objectives

The focus of this research was to develop an automatic programming system that receives a problem statement written in natural language and then generates the relevant source code. The generated source code should provide the desired functionality based on the problem statement.

In addition, the performance of several variants of LSTM networks, such as the Encoder-Decoder, Normal Sequential, and the combination of both of the models was examined via two experiments. The first experiment examined the performance of the combination of the Encoder-Decoder and the Normal Sequential model, whereas the second experiment examined the performance of the Normal Sequential model only. The Encoder-Decoder model captures the semantic representation of input whereas the Normal Sequential model predicts the next character given some previous characters as the seed. Based on these concepts, the first experiment tried to generate the final source code in which the decoded semantic representation became the seed for the Normal Sequential model, whereas the second experiment tried to generate the final source code in which some characters of a training sample became the seed.
