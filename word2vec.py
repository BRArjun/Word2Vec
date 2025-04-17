import glob
import random
import numpy as np
import os.path as op
import pickle
from utils.treebank import StanfordSentiment
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time

def softmax(x):
    orig_shape = x.shape
    if len(x.shape) > 1:
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        x = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    else:
        x = x - np.max(x)
        exp_x = np.exp(x)
        x = exp_x / np.sum(exp_x)
    assert x.shape == orig_shape
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(sigmoid_x):
    return sigmoid_x * (1 - sigmoid_x)

def gradient_checker(f, x):
    rndstate = random.getstate()
    random.setstate(rndstate)
    cost, grad = f(x)
    epsilon = 1e-4

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        i = it.multi_index

        x_minus = np.copy(x)
        x_minus[i] = x[i] - epsilon
        random.setstate(rndstate)
        f_minus = f(x_minus)[0]

        x_plus = np.copy(x)
        x_plus[i] = x[i] + epsilon
        random.setstate(rndstate)
        f_plus = f(x_plus)[0]

        numgrad = (f_plus - f_minus) / (2 * epsilon)
        reldiff = abs(numgrad - grad[i]) / max(1, abs(numgrad), abs(grad[i]))
        if reldiff > 1e-5:
            print("Gradient check failed.")
            print(f"First gradient error found at index {i}")
            print(f"Your gradient: {grad[i]} \t Numerical gradient: {numgrad}")
            return

        it.iternext()
    print("Gradient check passed!")

def normalizeRows(a):
    a = a / np.sqrt(np.sum(a ** 2, axis=1, keepdims=True))
    return a

def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    eachWordProb = softmax(np.dot(predicted, outputVectors.T))
    cost = -np.log(eachWordProb[target])
    eachWordProb[target] -= 1
    gradPred = np.dot(eachWordProb, outputVectors)
    grad = eachWordProb[:, np.newaxis] * predicted[np.newaxis, :]
    return cost, gradPred, grad

def getNegativeSamples(target, dataset, K):
    indices = [None] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices

def negSamplingCostAndGradient(predicted, target, outputVectors, dataset, K=10):
    indices = [target]
    indices.extend(getNegativeSamples(target, dataset, K))
    eachWordProb = np.dot(outputVectors, predicted)
    cost = -np.log(sigmoid(eachWordProb[target])) - np.sum(np.log(sigmoid(-eachWordProb[indices[1:]])))
    opposite_sign = (1 - sigmoid(-eachWordProb[indices[1:]]))
    gradPred = (sigmoid(eachWordProb[target]) - 1) * outputVectors[target] + \
               np.sum(opposite_sign[:, np.newaxis] * outputVectors[indices[1:]], axis=0)
    grad = np.zeros_like(outputVectors)
    grad[target] = (sigmoid(eachWordProb[target]) - 1) * predicted
    for k in indices[1:]:
        grad[k] += (1.0 - sigmoid(-np.dot(outputVectors[k], predicted))) * predicted
    return cost, gradPred, grad

def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)
    centerWord = tokens[currentWord]
    for contextWord in contextWords:
        target = tokens[contextWord]
        newCost, newGradPred, newGrad = word2vecCostAndGradient(
            inputVectors[centerWord], target, outputVectors, dataset)
        cost += newCost
        gradIn[centerWord] += newGradPred
        gradOut += newGrad
    return cost, gradIn, gradOut

def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
         dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)
    target = tokens[currentWord]
    centerWord = np.sum([inputVectors[tokens[contextWord]] for contextWord in contextWords], axis=0)
    cost, gradPred, gradOut = word2vecCostAndGradient(centerWord, target, outputVectors, dataset)
    for contextWord in contextWords:
        gradIn[tokens[contextWord]] += gradPred
    return cost, gradIn, gradOut

def load_saved_params():
    st = 0
    for f in glob.glob("saved_params_*.npy"):
        iter_num = int(op.splitext(op.basename(f))[0].split("_")[2])
        if iter_num > st:
            st = iter_num
    if st > 0:
        with open(f"saved_params_{st}.npy", "rb") as f:
            params = pickle.load(f)
            state = pickle.load(f)
        return st, params, state
    else:
        return st, None, None

def save_params(iter, params):
    with open(f"saved_params_{iter}.npy", "wb") as f:
        pickle.dump(params, f)
        pickle.dump(random.getstate(), f)

SAVE_PARAMS_EVERY = 5000

def sgd(f, x0, learning_rate, iterations, postprocessing=None, useSaved=False, PRINT_EVERY=10):
    ANNEAL_EVERY = 20000
    if useSaved:
        start_iter, oldx, state = load_saved_params()
        if start_iter > 0:
            x0 = oldx
            learning_rate *= 0.5 ** (start_iter // ANNEAL_EVERY)
        if state:
            random.setstate(state)
    else:
        start_iter = 0
    x = x0
    if not postprocessing:
        postprocessing = lambda x: x
    expcost = None
    for iter in range(start_iter + 1, iterations + 1):
        cost, grad = f(x)
        x -= learning_rate * grad
        x = postprocessing(x)
        if iter % PRINT_EVERY == 0:
            if not expcost:
                expcost = cost
            else:
                expcost = 0.95 * expcost + 0.05 * cost
            print(f"iter {iter}: {expcost}")
        if iter % SAVE_PARAMS_EVERY == 0 and useSaved:
            save_params(iter, x)
        if iter % ANNEAL_EVERY == 0:
            learning_rate *= 0.5
    return x

# Softmax and sigmoid test functions
def test_softmax():
    print("Running softmax tests...")
    test1 = softmax(np.array([[1, 2]]))
    ans1 = np.array([[0.26894142, 0.73105858]])
    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)

    test2 = softmax(np.array([[1001, 1002], [3, 4]]))
    ans2 = np.array([[0.26894142, 0.73105858], [0.26894142, 0.73105858]])
    assert np.allclose(test2, ans2, rtol=1e-05, atol=1e-06)

    test3 = softmax(np.array([[-1001, -1002]]))
    ans3 = np.array([[0.73105858, 0.26894142]])
    assert np.allclose(test3, ans3, rtol=1e-05, atol=1e-06)
    print("Passed!\n")

def test_sigmoid():
    print("Running sigmoid tests...")
    x = np.array([[1, 2], [-1, -2]])
    out = sigmoid(x)
    expected = 1 / (1 + np.exp(-x))
    assert np.allclose(out, expected), "Sigmoid failed!"
    print("Passed!\n")

