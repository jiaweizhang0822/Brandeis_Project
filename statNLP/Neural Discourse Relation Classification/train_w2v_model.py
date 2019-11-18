#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 21:55:21 2019

@author: jiawei
"""

from gensim.models import Word2Vec, KeyedVectors
import json
import string
from nltk.corpus import stopwords
import numpy as np
import datetime
import re

ARG1  = 'Arg1'
ARG2  = 'Arg2'
CONN  = 'Connective'
SENSE = 'Sense'
KEYS  = [ARG1, ARG2, CONN, SENSE]
TEXT  = 'RawText'
TYPE  = 'Type'
VEC   = '_vec'

def simple_tokenize(data):
    '''
    simple data preprocess: turn to lowercase; remove all punctuation;
    split on space; replace all number with string 'Number';
    '''
    stopword = stopwords.words('english')
    data = data.translate(str.maketrans('','',string.punctuation)).lower()
    data = data.split()
    data = [re.sub('\d', '', i) for i in data]
    data = [i if i!= '' else 'Number' for i in data]
    data = [i for i in data if i not in stopword]
    return data

def get_raw_train():
    '''get raw train data'''
    pdtb_file = open("data/train/relations.json", encoding='utf-8') 
    relations = [json.loads(x) for x in pdtb_file]
    return relations

def get_raw_dev():
    '''get raw dev data'''
    pdtb_file = open("data/dev/relations.json", encoding='utf-8') 
    relations = [json.loads(x) for x in pdtb_file]
    return relations

def get_train_corpus(relations):
    '''get relation data's corpus
    
    Args:
        relation: list
        
    Return:
        list, tokenized of relation
    '''
    corpus = [i[ARG1][TEXT] + " " + i[CONN][TEXT] + " " + i[ARG2][TEXT] for i in relations]
    tokenized_corpus = [simple_tokenize(i) for i in corpus]
    return tokenized_corpus

        
def get_selftrained_w2v_model(tokenized_corpus, name):
    '''
    get self-trained word2vec model, trained on training of train set's corpus
    
    Args:
        tokenized_corpus, see above
    
    Return:
        word2vec model
    '''
    # Set values for various parameters
    feature_size = 300    # Word vector dimensionality  
    window_context = 5          # Context window size                                                                                    
    min_word_count = 1   # Minimum word count                        
    #sample = 1e-3   # Downsample setting for frequent words
    print('Training word2vec model...')
    w2v_model = Word2Vec(tokenized_corpus, size = feature_size, workers = 4,
                          window = window_context, min_count = min_word_count,
                          iter = 50)
    w2v_model.save(name)
    print('Finished...')
    return w2v_model

def save_self_trained_models():
    tokenized_corpus = get_train_corpus(get_raw_train()) + get_train_corpus(get_raw_dev())
    
    get_selftrained_w2v_model(tokenized_corpus, "self_trained_w2v_on_train.model")

save_self_trained_models()

# self1 = "self_trained_w2v_on_train.model"
    
#start = datetime.datetime.now()
#save_self_trained_models()
#end = datetime.datetime.now()
#print (end-start)

def get_selftrained_wv_dict(model_name):
    '''
    load word2vec model from local
    save word's vector into a dict, add UNK (avg of vectors) and PAD (all 0)
    
    Return:
        a dict, with key denote words and value denote vector
    '''
    model = Word2Vec.load(model_name)
    wv_dict = {}
    for k, v in enumerate(model.wv.vocab):
        wv_dict[v] = model.wv[v]
    UNK_vec = sum(wv_dict.values()) / len(wv_dict)
    wv_dict['UNK'] = UNK_vec
    wv_dict['PAD'] = np.zeros(len(UNK_vec))
    return wv_dict

def get_glove_wv_dict(model_name):
    '''
    load glove word vector from local
    save word's vector into a dict, add UNK (avg of vectors) and PAD (all 0)
  
    Return:
        a dict, with key denote words and value denote vector
    '''
    wv_dict = {}
    f = open(model_name)
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        wv_dict[word] = vector
            
    f.close()
    wv_dict['UNK'] = sum(wv_dict.values()) / len(wv_dict)
    wv_dict['PAD'] = np.zeros(len(wv_dict['unk']))
    return wv_dict


    