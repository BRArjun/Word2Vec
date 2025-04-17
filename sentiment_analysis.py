#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import itertools

from utils.treebank import StanfordSentiment
import utils.glove as glove

from word2vec import load_saved_params, sgd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


def getArguments():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pretrained", dest="pretrained", action="store_true",
                       help="Use pretrained GloVe vectors.")
    group.add_argument("--yourvectors", dest="yourvectors", action="store_true",
                       help="Use your vectors from q3.")
    return parser.parse_args()


def getSentenceFeatures(tokens, wordVectors, sentence):
    sentVector = np.zeros((wordVectors.shape[1],))
    no_of_words = 0
    for word in sentence:
        if word in tokens:
            no_of_words += 1
            sentVector += wordVectors[tokens[word]]
    if no_of_words > 0:
        sentVector /= no_of_words
    return sentVector


def getRegularizationValues():
    values = [2 ** p for p in np.arange(0, 20, 2)] + [0.5 ** p for p in np.arange(0, 20, 2)]
    return sorted(values)


def chooseBestModel(results):
    return max(results, key=lambda x: x["dev"])


def accuracy(y, yhat):
    assert y.shape == yhat.shape
    return np.sum(y == yhat) * 100.0 / y.size


def plotRegVsAccuracy(regValues, results, filename):
    plt.plot(regValues, [x["train"] for x in results])
    plt.plot(regValues, [x["dev"] for x in results])
    plt.xscale('log')
    plt.xlabel("regularization")
    plt.ylabel("accuracy")
    plt.legend(['train', 'dev'], loc='upper left')
    plt.savefig(filename)


def outputConfusionMatrix(features, labels, clf, filename):
    pred = clf.predict(features)
    cm = confusion_matrix(labels, pred, labels=range(5))
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds)
    plt.colorbar()
    classes = ["- -", "-", "neut", "+", "+ +"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], ha="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(filename)


def outputPredictions(dataset, features, labels, clf, filename):
    pred = clf.predict(features)
    with open(filename, "w", encoding="utf-8") as f:
        print("True\tPredicted\tText", file=f)
        for i in range(len(dataset)):
            print(f"{labels[i]}\t{pred[i]}\t{' '.join(dataset[i][0])}", file=f)


def main(args):
    dataset = StanfordSentiment()
    tokens = dataset.tokens()
    nWords = len(tokens)

    if args.yourvectors:
        _, wordVectors, _ = load_saved_params()
        wordVectors = np.concatenate(
            (wordVectors[:nWords, :], wordVectors[nWords:, :]),
            axis=1)
    elif args.pretrained:
        wordVectors = glove.loadWordVectors(tokens)

    dimVectors = wordVectors.shape[1]

    # Train set
    trainset = dataset.getTrainSentences()
    trainFeatures = np.zeros((len(trainset), dimVectors))
    trainLabels = np.zeros((len(trainset),), dtype=np.int32)
    for i in range(len(trainset)):
        words, trainLabels[i] = trainset[i]
        trainFeatures[i, :] = getSentenceFeatures(tokens, wordVectors, words)

    # Dev set
    devset = dataset.getDevSentences()
    devFeatures = np.zeros((len(devset), dimVectors))
    devLabels = np.zeros((len(devset),), dtype=np.int32)
    for i in range(len(devset)):
        words, devLabels[i] = devset[i]
        devFeatures[i, :] = getSentenceFeatures(tokens, wordVectors, words)

    # Test set
    testset = dataset.getTestSentences()
    testFeatures = np.zeros((len(testset), dimVectors))
    testLabels = np.zeros((len(testset),), dtype=np.int32)
    for i in range(len(testset)):
        words, testLabels[i] = testset[i]
        testFeatures[i, :] = getSentenceFeatures(tokens, wordVectors, words)

    results = []
    regValues = getRegularizationValues()
    for reg in regValues:
        print(f"Training for reg={reg:.6f}")
        clf = LogisticRegression(C=1.0/(reg + 1e-12), max_iter=1000)
        clf.fit(trainFeatures, trainLabels)

        trainAccuracy = accuracy(trainLabels, clf.predict(trainFeatures))
        print(f"Train accuracy (%): {trainAccuracy:.2f}")

        devAccuracy = accuracy(devLabels, clf.predict(devFeatures))
        print(f"Dev accuracy (%): {devAccuracy:.2f}")

        testAccuracy = accuracy(testLabels, clf.predict(testFeatures))
        print(f"Test accuracy (%): {testAccuracy:.2f}")

        results.append({
            "reg": reg,
            "clf": clf,
            "train": trainAccuracy,
            "dev": devAccuracy,
            "test": testAccuracy
        })

    print("\n=== Recap ===")
    print("Reg\t\tTrain\tDev\tTest")
    for result in results:
        print("%.2E\t%.2f\t%.2f\t%.2f" % (
            result["reg"],
            result["train"],
            result["dev"],
            result["test"]))
    print()

    bestResult = chooseBestModel(results)
    print(f"Best regularization value: {bestResult['reg']:.2E}")
    print(f"Test accuracy (%): {bestResult['test']:.2f}")

    if args.pretrained:
        plotRegVsAccuracy(regValues, results, "q4_reg_v_acc.png")
        outputConfusionMatrix(devFeatures, devLabels, bestResult["clf"],
                              "q4_dev_conf.png")
        outputPredictions(devset, devFeatures, devLabels, bestResult["clf"],
                          "q4_dev_pred.txt")


if __name__ == "__main__":
    main(getArguments())

