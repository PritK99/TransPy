# Transformers: Attention is all you need

## Introduction

In order to improve the RNN architecture, which faces issues such as vanishing gradients, architectures such as GRUs and LSTMs were intorduced. Compared to RNN, these models could capture long term context. However, these are still sequential models. There is this notion of reccurence which acts as a bottleneck for computations. Furthermore, parallelization is not possible. 

The transformer architecture is based on ideas of attention and CNNs. The key concepts include:

* embeddings and positional encodings
* Self Attention
* Multi-Head Attention

## Embeddings and Positional Encodings

## Self Attention

For each word, we create a attention representation. Consider ```The cat jumped over the fence```. Hence our task is to compute values ```A<1>```, ```A<2>``` ... ```A<6>```. These calculations are done parallely, unlike the attention model of RNNs.

The equation for calculation of these attention values is:

<img src="../assets/Self-Formula.png" alt="Self-Formula">

The ```Q```, ```K```, ```V``` matrices here are simply embedding matrices of size ```(max_seq_len, embedding_dim)```.

### Idea of Key, Query, Value

<img src="../assets/Key-Query-Value.png" alt="Key-Query-Value">

Each of the input word is associated with a query, key and value pair. Thus ```A<1>``` has its own set of ```q<1>```, ```k<1>```, ```v<1>```.

Very similar to the idea of databases, query for each word is a question, how relevant is this word to context. This query is compared with keys of all words and likewise their values are extracted. The summation of these values give the attention representation for the given word with respect to all other words in the sentence.

Additionally, Self Attention does not require any new learnable parameters since it uses embeddings and positional encodings. Also, we use masks in Self Attention to avoid interaction of few words (for eg. PAD token).

## Multi-Head Attention

Multi-Head Attention is performing Self Attention multiple times. This allows us to compute a richer attention vector. Here, the intuition is that different heads would pose different questions and hence understand a different ascpect of input sequence.

<img src="../assets/Multi-Head Attention.jpg" alt="Multi Head Attention">

Unlike Self Attention, we use weight matrices ```Wq```, ```Wk```, ```Wv``` to compute ```Q```, ```K```, ```V```.

Additionally, we feed the entire input sequence to all the heads. However the embeddings that all these heads see are different. This is what leads them to understand different aspects of input sequence.

## Transformer Architecture

<img src = "../assets/Transformer-architecture.png" alt = "Transformer-Architecture">

## References

* <a href="https://github.com/hkproj/transformer-from-scratch-notes/tree/main">GitHub Link</a> to transformers-from-scratch-notes repository.
* ```Sequence Models``` course by deeplearning.ai on coursera.