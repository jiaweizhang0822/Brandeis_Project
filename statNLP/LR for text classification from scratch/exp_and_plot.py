#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 23:59:16 2019

@author: jiawei
"""

import maxent
from corpus import ReviewCorpus, BagOfWords, BagOfWordsBigram, BagOfWordsTrigram
from random import shuffle, seed
import matplotlib.pyplot as plt
import numpy as np
import datetime

#1. training data size exp
reviews = ReviewCorpus('yelp_reviews.json', document_class = BagOfWords)
seed(hash("reviews"))
shuffle(reviews)

dev, test = reviews[-2000:-1000], reviews[-1000:]
data_for_v = reviews[-22000:-2000]
train_size = [1000, 10000, 50000, 100000]
logit = maxent.MaxEnt()
logit.set_whole_vocabulary(data_for_v)

def train_size_exp(train_size_lst, dev, data_for_v):
    accuracy = []
    for train_size in train_size_lst:
        train = reviews[0:train_size]
        logit.train_sgd(train, dev, 0.0005, 100, 0.1, True)
        accuracy.append(logit.accuracy(dev))
    return accuracy, logit

exp1_accuracy, logit_exp1 = train_size_exp(train_size, dev, data_for_v)

def plot(exp_accuracy, exp_x, labels):
    figure_count=1
    plt.figure(figure_count)
    index = np.arange(len(exp_x))
    bar_width = 0.35
    rects1 = plt.bar(index, exp_accuracy, bar_width,
                     color='black',
                     label = labels
               )
     
    plt.xlabel('Training set size')
    plt.ylabel('Development set accuracy')
    plt.ylim(0.62,0.66)
    plt.xticks(index, exp_x)        
    plt.tight_layout()
    plt.legend()
    plt.show()

labels = "learning rate = .0005\nbatch size  = 100\nlambda = 0.1"
plot(exp1_accuracy, train_size, labels)

#2. minibatch data size exp
minibatch_size = [1, 10, 50, 100, 1000]
train_size = 50000
def minibatch_size_exp(train_size, minibatch_size_lst, dev, data_for_v):
    accuracy = []
    time = []
    for minibatch_size in minibatch_size_lst:
        start = datetime.datetime.now()
        train = reviews[0:train_size]
        logit.train_sgd(train, dev, 0.0005, minibatch_size, 0.1, True)        
        accuracy.append(logit.accuracy(dev))
        end = datetime.datetime.now()
        time.append(end-start)
    return accuracy, time, logit

exp2_accuracy, exp2_time, logit_exp2 = minibatch_size_exp(train_size, minibatch_size, dev, data_for_v)

def plot1(exp_accuracy, exp_time, exp_x, labels):

    figure_count=1
    plt.figure(figure_count)
    index = np.arange(len(exp_x))
    bar_width = 0.35
    rects1 = plt.bar(index, exp_accuracy, bar_width,
                     color='black',
                     label = labels
               )
    plt.xlabel('Minibatch size')
    plt.ylabel('Development set accuracy')
    plt.ylim(0.55,0.72)
    plt.xticks(index, exp_x)        
    plt.tight_layout()
    plt.legend()
    plt.show()

labels = "training size = 50000\nlearning rate = .0005\nlambda = 0.1"
plot1(exp2_accuracy, exp2_time, minibatch_size, labels)


#3. lambda exp
lambda_lst = [0.1, 0.5, 1, 10]
minibatch_size = 1000
def reg_exp(train_size, minibatch_size, dev, lambda_lst, data_for_v):
    accuracy = []
    for lambda_i in lambda_lst:
        train = reviews[0:train_size]
        logit.train_sgd(train, dev, 0.0005, minibatch_size, lambda_i, True)      
        accuracy.append(logit.accuracy(dev))
    return accuracy, logit

exp3_accuracy, logit_exp3 = reg_exp(train_size, minibatch_size, dev, lambda_lst, data_for_v)

def plot2(exp_accuracy, exp_x, labels):

    figure_count=1
    plt.figure(figure_count)
    index = np.arange(len(exp_x))
    bar_width = 0.35
    rects1 = plt.bar(index, exp_accuracy, bar_width,
                     color='black',
                     label = labels
               )
    plt.xlabel('Lambda')
    plt.ylabel('Development set accuracy')
    plt.ylim(0.5,0.8)
    plt.xticks(index, exp_x)        
    plt.tight_layout()
    plt.legend()
    plt.show()

labels = "training size = 50000\nminibatch size = 1000\nlearning rate = .0005"
plot2(exp3_accuracy, lambda_lst, labels)

#4 exp4 add bigram
reviews = ReviewCorpus('yelp_reviews.json', document_class = BagOfWordsBigram)
seed(hash("reviews"))
shuffle(reviews)

dev, test = reviews[-2000:-1000], reviews[-1000:]
data_for_v = reviews[-22000:-2000]

train_size = 50000
minibatch_size = 1000
lambda_l2 = 1

train = reviews[0:train_size]
logit = maxent.MaxEnt()
logit.set_whole_vocabulary(data_for_v)
logit.train_sgd(train, dev, 0.0005, minibatch_size, lambda_l2, True)

#5 exp4 add trigram
reviews = ReviewCorpus('yelp_reviews.json', document_class = BagOfWordsTrigram)
seed(hash("reviews"))
shuffle(reviews)

dev, test = reviews[-2000:-1000], reviews[-1000:]
data_for_v = reviews[-22000:-2000]

train_size = 50000
minibatch_size = 1000
lambda_l2 = 1

train = reviews[0:train_size]
logit = maxent.MaxEnt()
logit.set_whole_vocabulary(data_for_v)
logit.train_sgd(train, dev, 0.0005, minibatch_size, lambda_l2, True)
 
