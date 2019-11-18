#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 14:27:26 2019

@author: jiawei
"""
import numpy as np
import scipy.special
import json
from preprocessing import load_data
from preprocessing import get_all_sense, get_raw_train

def sigmoid(x):
    ''' sigmoid transformation of array x'''
    return (1 / (1 + np.exp(-x)))
def softmax(x):
    '''
    softmax transformation of matrix x alone columns
    Args:
        x: matrix with type = float
    '''
    y = x.copy()
    for i in range(x.shape[1]):
        y[:,i] = np.exp(x[:,i] - scipy.special.logsumexp(x[:,i]))   
    return y

class DRSClassifier_scratch(object):
    """TODO: Implement a FeedForward Neural Network for Discourse Relation Sense
      Classification using Tensorflow/Keras (tensorflow 2.0)
    """
    def __init__(self):
        self.w1 = 0
        self.w2 = 0
        self.w3 = 0
    
    def fwd_computation(self, x):
        """
        forward computation of 2 hidden layer neural netword
        
        Arg2:
            w1: weight matrix between input and h1
            shape of (#features, h1_neu)
            w2: weight matrix between h1 and h2
            shape of (h1_neu, h2_neu)
            w3: weight matrix between h2 and output
            shape of (h2_neu, h3_neu)
            x: batch of input, shape of (#features, n_obs)
            
        Return:
            s1(dot prod), z1(activation)
            s2(dot prod), z2(activation) 
            o(dot prod), yhat(activation)
        """
        
        s1 = self.w1.T@x
        z1 = sigmoid(s1)
        s2 = self.w2.T@z1
        z2 = sigmoid(s2)
        o = self.w3.T@z2
        yhat = softmax(o)
        return z1, z2, yhat
    
    
    def backpropagation(self, yhat, true_label, x, z1, z2):
        """
        return all necessary gradient
        
        Args:
            yhat: predict prob over all labels, shape (#categories, n_obs)
            true_label: one hot encoded, shape (#categories, n_obs)
            x: batch of input, shape of (#features, n_obs)
            z1: shape of (h1_neus, n_obs)
            z2: shape of (h2_neus, n_obs)
            w1: weight matrix between input and h1
            shape of (input_len, h1_neu)
            w2: weight matrix between h1 and h2
            shape of (h1_neu, h2_neu)
            w3: weight matrix between h2 and output
            shape of (h2_neu, h3_neu)
        """
        o_grad = yhat - true_label
    
        batch_size = yhat.shape[1]
        wts_Z2O_grad = np.zeros((batch_size, z2.shape[0], o_grad.shape[0]))
        for i in range(batch_size):
            wts_Z2O_grad[i] = np.atleast_2d(z2[:,i]).T@np.atleast_2d(o_grad[:,i])
        
        z2_grad = self.w3 @ o_grad 
        s2_grad = z2_grad * z2 * (1 - z2)
    
        wts_Z1S2_grad = np.zeros((batch_size, z1.shape[0], s2_grad.shape[0]))
        for i in range(batch_size):
            wts_Z1S2_grad[i] = np.atleast_2d(z1[:,i]).T@np.atleast_2d(s2_grad[:,i])
        
        z1_grad = self.w2 @ s2_grad 
        s1_grad = z1_grad * z1 * (1 - z1)
    
        wts_XS1_grad = np.zeros((batch_size, x.shape[0], s1_grad.shape[0]))
        for i in range(batch_size):
            wts_XS1_grad[i] = np.atleast_2d(x[:,i]).T@np.atleast_2d(s1_grad[:,i])
    
        return wts_Z2O_grad, wts_Z1S2_grad, wts_XS1_grad
    
    def chop_up(self, train_data, batch_size):
        '''
        chop data into minibatches, each size is batch_size
        
        Args:
            train_data: array, shape(n_obs, #features, len(word_vec))
        Return:
            list of array, each element is a batch
        '''
        ln = train_data.shape[0]
        minibatches = [train_data[x : x + batch_size] for x in range(0, ln - batch_size, batch_size)]
        last_batch = train_data[ (ln // batch_size) * batch_size : ]
        minibatches.append(last_batch)
        return minibatches  
    
        
    def get_feature_and_label(self,instances):
        """
        get feature and label from instances(flattened)
        
        Args:
            instances: list, each element is a dict
            
        Return:
            feature matirx: shape(n_obs, #features)
            labels: shape(n_obs, #categories)
        """
        rows = instances[0]['Arg1']['Vec'].shape[0] + instances[0]['Arg2']['Vec'].shape[0] + instances[0]['Connective']['Vec'].shape[0]
        cols = instances[0]['Arg1']['Vec'].shape[1]
        feature = np.zeros((len(instances), rows, cols))
        label = np.zeros((len(instances), 21))
        for i in range(len(instances)):
            a1 = instances[i]['Arg1']['Vec']
            a2 = instances[i]['Arg2']['Vec']
            c = instances[i]['Connective']['Vec']
            feature[i] = np.vstack((a1, a2, c))
            label[i] = instances[i]['SenseVec']
        feature_flat = np.reshape(feature, (feature.shape[0], feature.shape[1]*feature.shape[2]))
        return feature_flat, label
    
    def train_ffn(self, train_instances, dev_instances, first_neu, second_neu, learning_rate, batch_size, epochs):
        
        train_data, train_labels = self.get_feature_and_label(train_instances)
        dev_data, dev_labels = self.get_feature_and_label(dev_instances)
        
        train_data_batches = self.chop_up(train_data, batch_size)
        train_labels_batches = self.chop_up(train_labels, batch_size)
        input_len = train_data[0].shape[0]
        
        self.w1 = np.random.random((input_len, first_neu))
        self.w2 = np.random.random((first_neu, second_neu))
        self.w3 = np.random.random((second_neu, 21))
        count = 0
        # number of iteration times over whole train data
        for i in range(epochs):
            print('epochs', i)
            # number of batches to update
            ln = len(train_labels_batches)
            for j in range(ln):    
                print(j,'th batch...')
                x = train_data_batches[j].T
                z1, z2, yhat = self.fwd_computation(x)
                true_label = train_labels_batches[j].T
                w3_batch_grad, w2_batch_grad, w1_batch_grad = self.backpropagation(yhat, true_label, x, z1, z2)
                w3_grad = w3_batch_grad.sum(axis = 0)
                w2_grad = w2_batch_grad.sum(axis = 0)
                w1_grad = w1_batch_grad.sum(axis = 0)
                
                self.w3-= learning_rate * w3_grad
                self.w2-= learning_rate * w2_grad
                self.w1-= learning_rate * w1_grad
                count+= 1
                #train and dev loss and accuracy, for consideration of time, choose not to compute
                if count == -5:
                    _,_,train_yhat = self.fwd_computation(train_data.T)
                    train_loss = self.calculate_loss(train_yhat, train_labels.T)
                    print('train loss {0:6.2f}'.format(train_loss))
                    train_acc = self.accuracy(train_yhat, train_labels.T)
                    print('train accuracy {0:4.2f}'.format(train_acc))
                    _,_,dev_yhat = self.fwd_computation(dev_data.T)
                    dev_loss = self.calculate_loss(dev_yhat, dev_labels.T)
                    print('dev loss {0:6.2f}'.format(dev_loss))
                    dev_acc = self.accuracy(dev_yhat, dev_labels.T)
                    print('dev accuracy {0:4.2f}'.format(dev_acc))
                    count = 0
    
    
    def calculate_loss(self, yhat, y):      
        '''
        calculate loss 
        
        Args:
            yhat: predicted prob, shape(#categories, n_obs)
            y: shape(#categories, n_obs)
        Return:
            loss
        '''
        n_obs = y.shape[1]     
        loss = 0
        for i in range(n_obs):
            loss-= np.log(yhat[np.argmax(y[:,i]),i])
        return loss
      
    def predictin(self, train_data):
        _,_,_,_,_, yhat = self.fwd_computation(train_data)
        return yhat
    
    def accuracy(self, yhat, y):
        '''
        calculate accuracy
        
        Args:
            yhat: predicted prob, shape(#categories, n_obs)
            y: shape(#categories, n_obs)
        Return:
            accuracy
        '''
        n_obs = y.shape[1]
        acc = 0
        for i in range(n_obs):
            if np.argmax(yhat[:,i]) == np.argmax(y[:,i]):
                acc+= 1.
        acc/=  n_obs
        return acc
        
        
    def predict(self, test_instances, export_file = "./preds.json"):
        '''
        predict the 
        
        Args:
            test_instaces: list, element is dict
            w1, w2, w3
        '''
        test_data, test_labels = self.get_feature_and_label(test_instances)
        
        _,_, test_yhat = self.fwd_computation(test_data.T)
       
        sense2label = get_all_sense(get_raw_train())
        label2sense = dict([(idx, sense) for idx, sense in enumerate(sense2label)])
    
        acc = 0
        with open(export_file, 'w', encoding='utf-8') as fh:
            for i in range(len(test_instances)):
                # Include the following fields from the original dict
                arg1d = {'TokenList' : test_instances[i]['Arg1']['TokenList']}
                arg2d = {'TokenList' : test_instances[i]['Arg2']['TokenList']}
                connd = {'TokenList' : test_instances[i]['Connective']['TokenList']}
                docid = test_instances[i]['DocID']
                #sense = test_instances[i]['Sense']
                doctype = test_instances[i]['Type']
            
                # And the prediction
                sense = label2sense[test_yhat[:,i].argmax()]
                if sense == test_instances[i]['Sense']:
                    acc+=1
            
                docd = dict([('Arg1', arg1d), 
                         ('Arg2', arg2d), 
                         ('Connective', connd), 
                         ('DocID', docid),
                         ('Sense', [sense]),
                         ('Type', doctype)
                         ])
            
                fh.write("{}\n".format(json.dumps(docd)))        
        print("test acc is: ", acc/test_labels.shape[0])
                
    


