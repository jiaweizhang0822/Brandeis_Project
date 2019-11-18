#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""Entry point for the Program Assignment 2

Feel free to change/restructure the code below
"""
from model import DRSClassifier
from nn import DRSClassifier_scratch
from preprocessing import load_data, load_data_embed
import sys

__author__ = 'Jiawei Zhang'


def run_scorer(preds_file):
    """Automatically runs `scorer.py` on model predictions

    TODO: You don't need to use this code if you'd rather run `scorer.py`
      manually.

    Args:
        preds_file: str, path to model's prediction file
    """
    import os
    import sys
    import subprocess

    if not os.path.exists(preds_file):
        print("[!] Preds file `{}` doesn't exist in `run_scorer.py`".format(preds_file))
        sys.exit(-1)

    python = 'python3.7' # TODO: change this to your python command
    scorer = './scorer.py'
    gold = './data/test/relations.json'
    auto = preds_file
    command = "{} {} {} {}".format(python, scorer, gold, auto)

    print("Running scorer with command:", command)
    proc = subprocess.Popen(
      command, stdout = sys.stdout, stderr = sys.stderr, shell = True,
      universal_newlines = True
    )
    proc.wait()   

def main():
    # loads and preprocesses data. See `preprocessing.py`
    data = load_data(data_dir = './data')

    # trains a classifier on `train` and `dev` set. See `model.py`
    clf = DRSClassifier()
    clf.train(train_instances = data['train'], dev_instances = data['dev'])

    # output model predictions on `test` set
    preds_file = "./preds.json"
    clf.predict(data['test'], export_file = preds_file)

    # measure the accuracy of model predictions using `scorer.py`
    run_scorer(preds_file)

if __name__ == '__main__':
    embedding = sys.argv[1]
    model = sys.argv[2]
    preds_file = "./preds.json"
    clf = DRSClassifier()
    
    if embedding == 'word2vec':
        data = load_data("self_trained_w2v_on_train.model", glove = False)
        if model == 'keras_ffn':
            clf.train_ffn(train_instances = data['train'], dev_instances = data['dev'], batch_size=100, epochs = 5)
            clf.predict(data['test'], embedding = False, export_file = preds_file)
        elif model == 'cnn':
            clf.train_cnn(train_instances = data['train'], dev_instances = data['dev'],batch_size=100,epochs = 5)
            clf.predict(data['test'], embedding = False, export_file = preds_file)
        elif model == 'sractch_ffn':
            clf1 = DRSClassifier_scratch()
            clf1.train_ffn(train_instances = data['train'],  dev_instances = data['dev'], 
            first_neu = 256,  second_neu = 128,  learning_rate = 0.001, batch_size = 100, epochs = 1)
            clf1.predict(data['test'])
        
    
    #data = load_raw_data()
    elif embedding == 'glove':
        data = load_data("glove.6B.300d.txt",  glove = True)
        if model == 'keras_ffn':
            clf.train_ffn(train_instances = data['train'], dev_instances = data['dev'], batch_size=100, epochs = 5)
        elif model == 'cnn':
            clf.train_cnn(train_instances = data['train'], dev_instances = data['dev'],batch_size=100,epochs = 5)
        clf.predict(data['test'], embedding = False, export_file = preds_file)
        
    elif embedding == 'keras_embedding':
        data = load_data_embed()
        if model == 'keras_ffn':
            clf.train_embedding_ffn(train_instances = data['train'], dev_instances = data['dev'], batch_size=100, epochs = 5)
        elif model == 'cnn':
            clf.train_embedding_cnn(train_instances = data['train'], dev_instances = data['dev'],batch_size=100,epochs = 5)
        clf.predict(data['test'], embedding = True, export_file = preds_file)
      
        
    
    run_scorer(preds_file)

