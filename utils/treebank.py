#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import numpy as np
import os
import random

class StanfordSentiment:
    def __init__(self, path=None, tablesize=1000000):
        if not path:
            path = "utils/datasets/stanfordSentimentTreebank"

        self.path = path
        self.tablesize = tablesize

    def tokens(self):
        if hasattr(self, "_tokens") and self._tokens:
            return self._tokens

        tokens = dict()
        tokenfreq = dict()
        wordcount = 0
        revtokens = []
        idx = 0

        for sentence in self.sentences():
            for w in sentence:
                wordcount += 1
                if w not in tokens:
                    tokens[w] = idx
                    revtokens.append(w)
                    tokenfreq[w] = 1
                    idx += 1
                else:
                    tokenfreq[w] += 1

        tokens["UNK"] = idx
        revtokens.append("UNK")
        tokenfreq["UNK"] = 1
        wordcount += 1

        self._tokens = tokens
        self._tokenfreq = tokenfreq
        self._wordcount = wordcount
        self._revtokens = revtokens
        return self._tokens

    def sentences(self):
        if hasattr(self, "_sentences") and self._sentences:
            return self._sentences

        sentences = []
        with open(os.path.join(self.path, "datasetSentences.txt"), "r", encoding='utf-8') as f:
            first = True
            for line in f:
                if first:
                    first = False
                    continue
                splitted = line.strip().split()[1:]
                sentences.append([w.lower() for w in splitted])

        self._sentences = sentences
        self._sentlengths = np.array([len(s) for s in sentences])
        self._cumsentlen = np.cumsum(self._sentlengths)
        return self._sentences

    def numSentences(self):
        if hasattr(self, "_numSentences") and self._numSentences:
            return self._numSentences
        else:
            self._numSentences = len(self.sentences())
            return self._numSentences

    def allSentences(self):
        if hasattr(self, "_allsentences") and self._allsentences:
            return self._allsentences

        sentences = self.sentences()
        rejectProb = self.rejectProb()
        tokens = self.tokens()
        allsentences = [
            [w for w in s if rejectProb[tokens[w]] <= 0 or random.random() >= rejectProb[tokens[w]]]
            for s in sentences * 30
        ]
        allsentences = [s for s in allsentences if len(s) > 1]
        self._allsentences = allsentences
        return self._allsentences

    def getRandomContext(self, C=5):
        allsent = self.allSentences()
        sentID = random.randint(0, len(allsent) - 1)
        sent = allsent[sentID]
        wordID = random.randint(0, len(sent) - 1)

        context = sent[max(0, wordID - C):wordID]
        if wordID + 1 < len(sent):
            context += sent[wordID + 1:min(len(sent), wordID + C + 1)]

        centerword = sent[wordID]
        context = [w for w in context if w != centerword]

        if len(context) > 0:
            return centerword, context
        else:
            return self.getRandomContext(C)

    def sent_labels(self):
        if hasattr(self, "_sent_labels") and self._sent_labels:
            return self._sent_labels

        dictionary = dict()
        phrases = 0
        with open(os.path.join(self.path, "dictionary.txt"), "r", encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                splitted = line.split("|")
                dictionary[splitted[0].lower()] = int(splitted[1])
                phrases += 1

        labels = [0.0] * phrases
        with open(os.path.join(self.path, "sentiment_labels.txt"), "r", encoding='utf-8') as f:
            first = True
            for line in f:
                if first:
                    first = False
                    continue
                line = line.strip()
                if not line:
                    continue
                splitted = line.split("|")
                labels[int(splitted[0])] = float(splitted[1])

        sent_labels = [0.0] * self.numSentences()
        sentences = self.sentences()
        for i in range(self.numSentences()):
            sentence = sentences[i]
            full_sent = " ".join(sentence).replace('-lrb-', '(').replace('-rrb-', ')')
            try:
                sent_labels[i] = labels[dictionary[full_sent]]
            except KeyError:
                # Handle common encoding issues by normalizing text
                try:
                    # Try replacing common problematic characters
                    full_sent_normalized = full_sent.replace("ã©", "é").replace("ã¨", "è").replace("ã´", "ô").replace("ã", "é")
                    sent_labels[i] = labels[dictionary[full_sent_normalized]]
                except KeyError:
                    # If still can't find it, try a more aggressive normalization
                    try:
                        # Remove all non-ASCII characters
                        full_sent_ascii = ''.join(c for c in full_sent if ord(c) < 128)
                        sent_labels[i] = labels[dictionary[full_sent_ascii]]
                    except KeyError:
                        # If all else fails, assign a neutral sentiment
                        print(f"Warning: Sentence not found in dictionary: {full_sent}")
                        sent_labels[i] = 0.5  # Neutral sentiment

        self._sent_labels = sent_labels
        return self._sent_labels

    def dataset_split(self):
        if hasattr(self, "_split") and self._split:
            return self._split

        split = [[] for _ in range(3)]
        with open(os.path.join(self.path, "datasetSplit.txt"), "r", encoding='utf-8') as f:
            first = True
            for line in f:
                if first:
                    first = False
                    continue
                splitted = line.strip().split(",")
                split[int(splitted[1]) - 1].append(int(splitted[0]) - 1)

        self._split = split
        return self._split

    def getRandomTrainSentence(self):
        split = self.dataset_split()
        sentId = split[0][random.randint(0, len(split[0]) - 1)]
        return self.sentences()[sentId], self.categorify(self.sent_labels()[sentId])

    def categorify(self, label):
        if label <= 0.2:
            return 0
        elif label <= 0.4:
            return 1
        elif label <= 0.6:
            return 2
        elif label <= 0.8:
            return 3
        else:
            return 4

    def getDevSentences(self):
        return self.getSplitSentences(2)

    def getTestSentences(self):
        return self.getSplitSentences(1)

    def getTrainSentences(self):
        return self.getSplitSentences(0)

    def getSplitSentences(self, split=0):
        ds_split = self.dataset_split()
        results = []
        for i in ds_split[split]:
            try:
                results.append((self.sentences()[i], self.categorify(self.sent_labels()[i])))
            except Exception as e:
                print(f"Warning: Skipping sentence {i} due to error: {e}")
                continue
        return results

    def sampleTable(self):
        if hasattr(self, '_sampleTable') and self._sampleTable is not None:
            return self._sampleTable

        nTokens = len(self.tokens())
        samplingFreq = np.zeros((nTokens,))
        self.allSentences()
        for i in range(nTokens):
            w = self._revtokens[i]
            if w in self._tokenfreq:
                freq = float(self._tokenfreq[w])
                freq = freq ** 0.75
            else:
                freq = 0.0
            samplingFreq[i] = freq

        samplingFreq /= np.sum(samplingFreq)
        samplingFreq = np.cumsum(samplingFreq) * self.tablesize

        self._sampleTable = [0] * self.tablesize
        j = 0
        for i in range(self.tablesize):
            while i > samplingFreq[j]:
                j += 1
            self._sampleTable[i] = j

        return self._sampleTable

    def rejectProb(self):
        if hasattr(self, '_rejectProb') and self._rejectProb is not None:
            return self._rejectProb

        threshold = 1e-5 * self._wordcount
        nTokens = len(self.tokens())
        rejectProb = np.zeros((nTokens,))
        for i in range(nTokens):
            w = self._revtokens[i]
            freq = float(self._tokenfreq[w])
            rejectProb[i] = max(0, 1 - np.sqrt(threshold / freq))

        self._rejectProb = rejectProb
        return self._rejectProb

    def sampleTokenIdx(self):
        return self.sampleTable()[random.randint(0, self.tablesize - 1)]
