#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""Data Loader/Pre-processor Functions

Feel free to change/restructure the code below
"""

import json
import os
import train_w2v_model as twm
import numpy as np
import datetime
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences


__author__ = 'Jiawei Zhang'


"""Useful constants when processing `relations.json`"""
ARG1  = 'Arg1'
ARG2  = 'Arg2'
CONN  = 'Connective'
SENSE = 'Sense'
TYPE  = 'Type'
DOCID = 'DocID'
ID    = 'ID'
KEYS  = [ARG1, ARG2, CONN, SENSE]
TEXT  = 'RawText'
VEC   = 'Vec'
TOK   = 'Token'

def preprocess(rels):
    """Tokenize ARG1, ARG2, CONN as key TOK, take first sense as label

    Args:
      rels: list, each element is a dict (json format)

    Returns:
      see `featurize` above
    """

    for rel in rels:
        for key in KEYS:
            if key in [ARG1, ARG2, CONN]:
                # for `Arg1`, `Arg2`, `Connective`, we only keep tokens of `RawText`
                rel[key][TOK] = twm.simple_tokenize(rel[key][TEXT])

            elif key == SENSE:
                # `Sense` is the target label. For relations with multiple senses, we
                # assume (for simplicity) that the first label is the gold standard.
                rel[key] = rel[key][0]

    return rels

def featurize(rels, wv_dict):
    """Featurizes a relation dict into feature vector （use word2vec embedding here）
    
    TODO: `rel` is a dict object for a single relation in `relations.json`, where
    `Arg1`, `Arg2`, `Connective` and `Sense` are all strings (`Conn` may be an
    empty string). Implement a featurization function that transforms this into
    a feature vector. You may use word embeddings.

    Args:
        rel: dict, stand for a single obs
        wv_dict, word vector dict
      
    Returns:
        add word vec (arg1, arg2, conn) to the dict
    """

    #truncate args for first xx words
    arg_keep_word = 100
    conn_keep_word = 1
    
    for rel in rels:
        '''
        k = ARG1
        ln = len(rel[k][TOK])
        if  ln >= arg_keep_word:
            temp = rel[k][TOK][ln - arg_keep_word : ]
            rel[k][VEC] = [wv_dict[i] if i in wv_dict else wv_dict['UNK'] for i in temp]
        else:
            rel[k][VEC] = [wv_dict[i] if i in wv_dict else wv_dict['UNK'] for i in rel[k][TOK]]
            for j in range(len(rel[k][VEC]), arg_keep_word):
                rel[k][VEC].insert(0, wv_dict['PAD'])
        '''
        
        k = ARG1
        ln = len(rel[k][TOK])
        if ln >= arg_keep_word:
            temp = rel[k][TOK][0 : arg_keep_word]
            rel[k][VEC] = [wv_dict[i] if i in wv_dict else wv_dict['UNK'] for i in temp]
        else:
            rel[k][VEC] = [wv_dict[i] if i in wv_dict else wv_dict['UNK'] for i in rel[k][TOK]]
            for j in range(ln, arg_keep_word):
                rel[k][VEC].append(wv_dict['PAD'])
        
        k = ARG2
        ln = len(rel[k][TOK])
        if ln >= arg_keep_word:
            temp = rel[k][TOK][0 : arg_keep_word]
            rel[k][VEC] = [wv_dict[i] if i in wv_dict else wv_dict['UNK'] for i in temp]
        else:
            rel[k][VEC] = [wv_dict[i] if i in wv_dict else wv_dict['UNK'] for i in rel[k][TOK]]
            for j in range(ln, arg_keep_word):
                rel[k][VEC].append(wv_dict['PAD'])
    
        #truncate CONN for first xx words
        k = CONN
        ln = len(rel[k][TOK])
        if ln >= conn_keep_word:
            temp = rel[k][TOK][0 : conn_keep_word]
            rel[k][VEC] = [wv_dict[i] if i in wv_dict else wv_dict['UNK'] for i in temp]
        else:
            rel[k][VEC] = [wv_dict[i] if i in wv_dict else wv_dict['UNK'] for i in rel[k][TOK]]
            for j in range(ln, conn_keep_word):
                rel[k][VEC].append(wv_dict['PAD'])
                
        for k in [ARG1, ARG2, CONN]:
            rel[k][VEC] = np.array(rel[k][VEC])
    
    return rels

def get_raw_train():
    '''get raw train data'''
    pdtb_file = open("data/train/relations.json", encoding='utf-8') 
    relations = [json.loads(x) for x in pdtb_file]
    return relations

def get_all_sense(raw_train):
    '''
    get all sense labels
    Args:
        raw_train, list, all raw data
    Return:
        dict: key - labeles; value - index
    '''
    all_sense = set()
    for i in range(len(raw_train)):
        all_sense.add(raw_train[i][SENSE][0])       
    all_sense_dict = dict([(label, idx) for idx, label in enumerate(all_sense)])
    return all_sense_dict

def encode_sense(rels):
    '''
    encode sense through one hot
    
    Args:
        list, total relations
    
    Return:
        relations with encoded sense
    '''
    raw_train = get_raw_train() 
    all_sense_dict = get_all_sense(raw_train)
    
    for i in range(len(rels)):
        rels[i][SENSE + VEC] = np.zeros(len(all_sense_dict))
        rels[i][SENSE + VEC][all_sense_dict[rels[i][SENSE]]] += 1
        
    return rels

def load_relations(folder_path, wv_dict):
    """Loads a single `relations.json` file

    Args:
      data_file: str, path to a single data file

    Returns:
      list, where each item is of type dict
    """
    rel_path = os.path.join(folder_path, "relations.json")
    assert os.path.exists(rel_path), \
    "{} does not exist in `load_relations.py".format(rel_path)

    pdtb_file = open(rel_path, encoding='utf-8')
    rels = [json.loads(x) for x in pdtb_file]
    rels = preprocess(rels)
    rels = featurize(rels, wv_dict)
    rels = encode_sense(rels)
    return rels

def load_data(model_name, glove, data_dir = './data'):
    """Loads all data in `data_dir` as a dict
    Each of `dev`, `train` and `test` contains (1) `raw` folder (2)
    `relations.json`. We don't need to worry about `raw` folder, and instead
    focus on `relations.json` which contains all the information we need for our
    classification task.

    Args:
    data_dir: str, the root directory of all data

    Returns:
    dict, where the keys are: `dev`, `train` and `test` and the values are lists
      of relations data in `relations.json`
    """
    
    assert os.path.exists(data_dir), "`data_dir` does not exist in `load_data`"

    data = {}
    if glove:
        wv_dict = twm.get_glove_wv_dict(model_name)
    else:
        wv_dict = twm.get_selftrained_wv_dict(model_name)
    
    for folder in os.listdir(data_dir):
        #no hidden files
        if folder[0] != '.':
            print("Loading", folder)
            start = datetime.datetime.now()
            folder_path = os.path.join(data_dir, folder)
            data[folder] = load_relations(folder_path, wv_dict)
            end = datetime.datetime.now()
            print ("Loading time: ", end-start)

    return data

def load_data_embed(data_dir = './data'):
    '''
    load raw data, for keras embedding
    
    Return:
        dict: train : train_relation (list), etc.
    '''
    assert os.path.exists(data_dir), "`data_dir` does not exist in `load_data`"

    data = {}
  
    for folder in os.listdir(data_dir):
        #no hidden files
        if folder[0] != '.':
            print("Loading", folder)
            start = datetime.datetime.now()
            folder_path = os.path.join(data_dir, folder)
            rel_path = os.path.join(folder_path, "relations.json")
            pdtb_file = open(rel_path, encoding='utf-8')
            rels = [json.loads(x) for x in pdtb_file]
            for rel in rels:
                rel[SENSE] = rel[SENSE][0]
            rels = encode_sense(rels)
            data[folder] = rels
            end = datetime.datetime.now()
            print ("Loading time: ", end-start)
    
    return data
