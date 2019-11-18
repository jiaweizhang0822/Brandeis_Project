#! /usr/bin/python3
# -*- coding: utf-8 -*-


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Flatten, Dense, Dropout, Activation, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Embedding
#from tensorflow.keras import backend
import numpy as np

import json
from preprocessing import get_all_sense, get_raw_train

max_length = 201

"""Discourse Relation Sense Classifier

Feel free to change/restructure the code below
"""

__author__ = 'Jiawei Zhang'


class DRSClassifier(object):
    """TODO: Implement a FeedForward Neural Network for Discourse Relation Sense
      Classification using Tensorflow/Keras (tensorflow 2.0)
    """
    def __init__(self):
        self.build()
        self.model = None

    def build(self):
    #TODO: Build your neural network here
        pass
    
    def get_feature_and_label(self, instances):
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
        return feature, label
    
    def get_feature_and_label_embed(self, instances, vocab_size, max_length):
        a1 = [one_hot(i['Arg1']['RawText'], vocab_size) for i in instances]
        a2 = [one_hot(i['Arg2']['RawText'], vocab_size) for i in instances]
        c =  [one_hot(i['Connective']['RawText'], vocab_size) for i in instances]
        a1 = pad_sequences(a1, maxlen = int ((max_length-1)/2), padding = 'post')
        a2 = pad_sequences(a1, maxlen = int ((max_length-1)/2), padding = 'post')
        c = pad_sequences(c, maxlen = 1, padding = 'post')
        feature = np.vstack((a1.T, a2.T, c.T))

        labels = [i['SenseVec'] for i in instances]
        return feature.T, np.array(labels)
    
    def train_embedding_ffn(self, train_instances, dev_instances, batch_size = 10,
              epochs = 5):
        
        vocab_size = 36000
        train_data, train_labels = self.get_feature_and_label_embed(
                train_instances, vocab_size = vocab_size, max_length = max_length)
        dev_data, dev_labels = self.get_feature_and_label_embed(
                dev_instances, vocab_size = vocab_size, max_length = max_length)
        
        model = Sequential([
                Embedding(input_dim = vocab_size, output_dim = 300, 
                          input_length = max_length),
                Flatten(),
                Dense(256, activation='relu'),
                Dense(128, activation='relu'),
                Dense(21, activation='softmax')
                ])
        model.summary()

        model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
            loss = 'categorical_crossentropy',
            metrics = ['accuracy']
        ) 

        # 4. train model
        model.fit(train_data, train_labels, batch_size = batch_size, epochs = epochs,
                  validation_data = (dev_data, dev_labels))
        
        # 5. evaluate with trained model
        self.model = model

    def train_embedding_cnn(self, train_instances, dev_instances, batch_size = 10,
              epochs = 5):
        
        vocab_size = 36000
        train_data, train_labels = self.get_feature_and_label_embed(
                train_instances, vocab_size = vocab_size, max_length = max_length)
        dev_data, dev_labels = self.get_feature_and_label_embed(
                dev_instances, vocab_size = vocab_size, max_length = max_length)
        
        model = Sequential([
                Embedding(input_dim = vocab_size, output_dim = 300, 
                          input_length = max_length),
                Conv1D(50, 3, activation = 'relu'),
                GlobalMaxPooling1D(), 
                Dense(128, activation = 'relu'),
                Dense(21, activation = 'softmax')
                ])
        model.summary()

        model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
            loss = 'categorical_crossentropy',
            metrics = ['accuracy']
        ) 

        # 4. train model
        model.fit(train_data, train_labels, batch_size = batch_size, epochs = epochs,
                  validation_data = (dev_data, dev_labels))
        
        # 5. evaluate with trained model
        self.model = model
    
    
    def train_ffn(self, train_instances, dev_instances, batch_size = 10,
              epochs = 5):
        """TODO: Train the classifier on `train_instances` while evaluating
        periodically on `dev_instances`

        Args:
        train_instances: list of relation dict
        dev_instances: list of relation dict
        """        
        train_data, train_labels = self.get_feature_and_label(train_instances)
        dev_data, dev_labels = self.get_feature_and_label(dev_instances)
        
        
        input_shape = train_data[0].shape
        #3 layer FNN model
        model = Sequential([
                Flatten(input_shape= input_shape),
                Dense(256, activation='relu'),
                #Dropout(0.5),
                Dense(128, activation='relu'),
                #Dropout(0.5),
                Dense(21, activation='softmax')
                ])
        model.summary()

        model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
        # default leanring rate of 0.001, see https://keras.io/optimizers/
            #optimizer='adam',
            loss = 'categorical_crossentropy',
            metrics = ['accuracy']
        ) 

        # 4. train model
        model.fit(train_data, train_labels, batch_size = batch_size, epochs = epochs,
                  validation_data = (dev_data, dev_labels))
        
        # 5. evaluate with trained model
        self.model = model


    def train_cnn(self, train_instances, dev_instances, batch_size = 10,
              epochs = 5):
        train_data, train_labels = self.get_feature_and_label(train_instances)
        dev_data, dev_labels = self.get_feature_and_label(dev_instances)
        
        
        input_shape = train_data[0].shape
        model = Sequential([
                Conv1D(50, 3, activation = 'relu', input_shape = input_shape),
                GlobalMaxPooling1D(),
                Dense(256, activation = 'relu'),
                Dense(128, activation = 'relu'),
                Dense(21, activation = 'softmax')
                ])
        model.summary()
        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
                #optimizer = 'adam',
                      loss = 'categorical_crossentropy',
                      metrics = ['accuracy'])
        
        model.fit(train_data, train_labels, batch_size = batch_size, epochs = epochs, 
                  validation_data = (dev_data, dev_labels))
        self.model = model
        

    def predict(self, test_instances, embedding = True, export_file="./preds.json"):
        """TODO: Given a trained model, make predictions on `instances` and export
        predictions to a json file

        Args:
            instances: list
            export_file: str, where to save your model's predictions on `instances`

        Returns:
            
            """
        if embedding:
            test_data, test_labels = self.get_feature_and_label_embed(
                    test_instances, vocab_size = 36000, max_length = max_length)
        else:
            test_data, test_labels = self.get_feature_and_label(test_instances)
            
        preds = self.model.predict(test_data)
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
                sense = label2sense[preds[i].argmax()]
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
                
        
        
        
        
        
