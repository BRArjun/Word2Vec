#!/usr/bin/env python3

import random
import numpy as np
from utils.treebank import StanfordSentiment
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from word2vec import *

# Load the Stanford Sentiment Treebank dataset
dataset = StanfordSentiment()
tokens = dataset.tokens()
nWords = len(tokens)

# Random initialization
dimVectors = 50
C = 5
random.seed(314159)
wordVectors = np.random.randn(2*nWords, dimVectors) / np.sqrt(dimVectors)
wordVectors0 = wordVectors.copy()

# Define cost and gradient function for SGD
def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C, word2vecCostAndGradient=softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    centerWordVectors = wordVectors[:nWords,:]
    outsideVectors = wordVectors[nWords:,:]
    for i in range(batchsize):
        centerWord, context = dataset.getRandomContext(C)
        c, gin, gout = word2vecModel(centerWord, C, context, tokens, centerWordVectors, outsideVectors, dataset, word2vecCostAndGradient)
        cost += c / batchsize
        grad[:nWords, :] += gin / batchsize
        grad[nWords:, :] += gout / batchsize
    return cost, grad

# Train word vectors using Skip-gram
print("=== Training word vectors using Skip-gram ===")
start_time = time.time()
wordVectors = sgd(
    lambda vec: word2vec_sgd_wrapper(skipgram, tokens, vec, dataset, C, negSamplingCostAndGradient),
    wordVectors, 0.3, 40000, None, True, PRINT_EVERY=1000
)
print("Training took %d seconds" % (time.time() - start_time))

# Save the word vectors
save_params(40000, wordVectors)
print("Word vectors saved to saved_params_40000.npy")
