import re
from collections import Counter

import numpy
import torch
from torch.utils.data import TensorDataset, DataLoader

import pickle
from .partitioner import Partition
import os.path

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

import torch
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

tokenizer = get_tokenizer("basic_english")

seq_length = 800
def pad_features(news, seq_length):
    features = numpy.zeros((len(news), seq_length), dtype=int)
    for i, row in enumerate(news):
        features[i, -len(row):] = numpy.array(row)[:seq_length]
    return features

def tokenize(train_array, test_array, backdoor_train= None, backdoor_test= None):
    """
    Tokenizes the given train and test data
    @param train_array: the training array
    @param test_array: the testing array
    @return: The vocab used, the tokenized train array, the tokenized test array
    """
    # get all the words in the input
    words = []
    for r in train_array:
        words.extend(r)

    for r in test_array:
        words.extend(r)

    if not backdoor_train is None and not backdoor_test is None:
        for r in backdoor_train:
            words.extend(r)

        for r in test_array:
            words.extend(r)

    ## Build a dictionary that maps words to integers
    counts = Counter(words)
    vocab = sorted(counts, key=counts.get, reverse=True)


    return_train_array = tokenize_array(train_array, vocab)

    return_test_array = tokenize_array(test_array, vocab)


    if backdoor_train is None or backdoor_test is None:
        return vocab, return_train_array, return_test_array

    return_backdoor_train_array = tokenize_array(backdoor_train, vocab)

    return_backdoor_test_array = tokenize_array(backdoor_test, vocab)

    return vocab, return_train_array, return_test_array, return_backdoor_train_array, return_backdoor_test_array

def tokenize_array(array, vocab):
    vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}
    return_array = []
    ## tokenize the train array
    for i in range(len(array)):
        return_array.append([vocab_to_int[word] for word in array[i]])
    return return_array


stem = PorterStemmer()

def clean(text):
    return text
    text = text.lower()
    text = re.sub('[^a-zA-Z0-9]',' ',text)
    text = text.split()
    text = [ i for i in text if i not in stopwords.words('english')]
    text = [stem.stem(i) for i in text]
    return ' '.join(text)

class AGNewsData:
    def __init__(self):
        # self.dataDir = dataPath
        self.bs = 80

    def buildDataset(self, backdoor=None, conf=None):

        # # read data from text files
        # X_train = np.loadtxt(fname=self.dataDir + 'sent140_{}_{}_trainX.np'.format(fractionOfTrain, th),
        #                      delimiter=",").astype(int)
        # Y_train = np.loadtxt(fname=self.dataDir + 'sent140_{}_{}_trainY.np'.format(fractionOfTrain, th),
        #                      delimiter=",").astype(int)
        # n = len(X_train)
        # # n = n - n%bs
        #
        # self.X_train = X_train[:n]
        # self.Y_train = Y_train[:n]
        # print(X_train.shape, Y_train.shape)
        #
        # X_test = np.loadtxt(fname=self.dataDir + 'sent140_{}_{}_testX.np'.format(fractionOfTrain, th),
        #                     delimiter=",").astype(int)
        # Y_test = np.loadtxt(fname=self.dataDir + 'sent140_{}_{}_testY.np'.format(fractionOfTrain, th),
        #                     delimiter=",").astype(int)
        #
        #

        train_iter = iter(AG_NEWS(split="train"))
        train_inputs =  []
        train_labels = []
        backdoor_train_inputs = []
        backdoor_test_inputs = []
        backdoor_train_labels = []
        backdoor_test_labels = []
        i = 0
        total_good = conf['totalUsers'] * conf["numIIDUsers"]
        print("building normal and backdoor data")
        for label, news in train_iter:
            if i <= total_good + 1600:
                train_inputs.append(clean(news))
                train_labels.append(label)

            elif not backdoor is None:
                if i <= total_good + 3200:
                    backdoor_train_inputs.append("Placeholder " + clean(news))
                    backdoor_train_labels.append(4)

                elif i <= total_good + 4200:
                    backdoor_test_inputs.append("Placeholder " + clean(news))
                    backdoor_test_labels.append(4)
            i += 1

        self.size_training = len(train_inputs)

        train_labels = numpy.array(train_labels)

        test_iter = iter(AG_NEWS(split="test"))
        test_inputs = []
        test_labels = []
        for label, news in test_iter:
            test_inputs.append(clean(news))
            test_labels.append(label)


        test_labels = numpy.array(test_labels)

        if not backdoor is None:
            self.vocab, train_inputs, test_inputs, backdoor_train_inputs, backdoor_test_inputs =\
                tokenize(train_inputs, test_inputs, backdoor_train_inputs, backdoor_test_inputs)
        else:
            self.vocab, train_inputs, test_inputs  = tokenize(train_inputs, test_inputs)

        train_inputs = pad_features(train_inputs, seq_length=seq_length)
        test_inputs = pad_features(test_inputs, seq_length=seq_length)

        backdoor_train_inputs = pad_features(backdoor_train_inputs, seq_length=seq_length)
        backdoor_test_inputs = pad_features(backdoor_test_inputs, seq_length=seq_length)

        train_inputs = numpy.array(train_inputs)
        test_inputs = numpy.array(test_inputs)


        total_users = conf['totalUsers']

        samples_per_user = conf["numIIDUsers"]
        ensure_total_samples = samples_per_user * total_users
        n = ensure_total_samples

        ## add extra for backdoor
        input_train_res = train_inputs[n:n + 2000]
        label_train_res = train_labels[n:n + 2000]
        ## ensure there is enough data for the backdoor
        # assert len(input_train_res) > 800
        # print('reserved samples ', len(X_train_res))

        train_inputs = train_inputs[:n]
        train_labels = train_labels[:n]

        # backdoor data
        if not backdoor is None:
            numEdgePtsAdv = conf['numEdgePtsAdv']
            advPts = 800

            Xb_train_adv = backdoor_train_inputs[:numEdgePtsAdv]
            Yb_train_adv = backdoor_train_labels[:numEdgePtsAdv]

            Xb_train = numpy.vstack((input_train_res[:advPts - numEdgePtsAdv], Xb_train_adv))
            Yb_train = numpy.concatenate((label_train_res[:advPts - numEdgePtsAdv], Yb_train_adv))

            n = len(backdoor_test_inputs)
            n = n - n % 20
            Xb_test = backdoor_test_inputs[:n]
            Yb_test = backdoor_test_labels[:n]

            # put edge points in good users
            numEdgePtsGood = conf['numEdgePtsGood']
            print('numEdgePtsGood,', numEdgePtsGood)

            print('lengths at adv ', len(Xb_train), len(Yb_train), len(Xb_test), len(Yb_test))

            self.backdoorTrainData = TensorDataset(torch.from_numpy(Xb_train), torch.from_numpy(Yb_train))
            self.backdoorTestData = TensorDataset(torch.from_numpy(Xb_test), torch.from_numpy(numpy.array(Yb_test)))


        print('total test ', len(test_inputs))
        m = len(test_inputs) - len(test_inputs) % self.bs
        self.X_test = test_inputs[:m]
        self.Y_test = test_labels[:m]
        self.X_train = train_inputs
        self.Y_train = train_labels

        print('final x train shape,', self.X_train.shape, self.Y_train.shape)

        self.trainData = TensorDataset(torch.from_numpy(self.X_train), torch.from_numpy(self.Y_train))
        self.testData = TensorDataset(torch.from_numpy(self.X_test), torch.from_numpy(self.Y_test))



        self.vocabSize = len(self.vocab)

    def getTrainDataForUser(self, userId):
        return self.lstParts[userId]

    def partitionTrainData(self, partitionType, numParts,conf=None):
        partitioner = Partition()

        if (partitionType == 'iid'):
            self.lstParts = partitioner.iidParts(self.trainData, numParts)
        elif (partitionType == 'hetero-dir'):
            self.lstParts = partitioner.hetero_dir_partition(self.trainData,self.Y_train,conf['totalUsers'], 4,
                                                             conf['beta'], conf["minRequiredSize"])
        elif (partitionType == 'natural'):
            pass
        else:
            raise ('{} partitioning not defined for this dataset'.format(partitionType))







