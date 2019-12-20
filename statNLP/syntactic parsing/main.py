#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""Entry Point

This project implements `Grammar as a Foreign Language` by Vinyals et al. (2014)
https://arxiv.org/abs/1412.7449

Feel free to change any part of this code
"""
import argparse
import os
import pickle
import time

from parser import Parser
from processor import postprocess, Processor

__author__ = 'Jiawei Zhang'

argparser = argparse.ArgumentParser("PA4 Argparser")

# paths
argparser.add_argument(
    '--data_dir', default='./data', help='path to data directory')
argparser.add_argument(
    '--evalb_dir', default='./EVALB', help='path to EVALB directory')

# preprocessing flags
argparser.add_argument(
    '--enc_max_len', default=-1, type=int,
    help='how many encoder tokens to keep. -1 for max length from corpus')
argparser.add_argument(
    '--dec_max_len', default=-1, type=int,
    help='how many decoder tokens to keep. -1 for max length from corpus')
argparser.add_argument(
    '--do_reverse', action='store_true', help='whether to reverse sents')
argparser.add_argument(
    '--do_lower', action='store_true', help='whether to lower-case sents')

# model flags
argparser.add_argument(
    '--emb_dim', default=64, type=int, help='length of embedding feature vector')
argparser.add_argument(
    '--num_layers', default=1, type=int,
    help='number of LSTM layers in encoder and decoder')
argparser.add_argument(
    '--hidden_dim', default=64, type=int, help='number of LSTM hidden units')
argparser.add_argument(
    '--attention', type=str.lower, default='', choices=['', 'luong', 'bahdanau'],
    help='which attention to use. \'\' for no attention')

# experiment flags
argparser.add_argument(
    '--batch_size', default=64, type=int, help='size of each mini batch')
argparser.add_argument(
    '--epochs', default=1, type=int, help='number of training iterations')

# misc
argparser.add_argument(
    '--verbose', action='store_true', help='whether to print each prediction')


def run_evalb(evalb_dir, gold_path, pred_path):
    """executes evalb automatically

  Assumed that `EVALB` is installed through `make` command
  """
    import sys
    import subprocess

    if not os.path.exists(pred_path):
        print(
            "[!] Preds file `{}` doesn't exist in `run_scorer.py`".format(pred_path))
        sys.exit(-1)

    evalb = os.path.join(evalb_dir, 'evalb')
    error_flag = '-e'
    num_error_trees = 10000  # arbitrarily big
    command = "{} {} {} {} {}".format(
        evalb, error_flag, num_error_trees, gold_path, pred_path)

    print("Running EVALB with command:", command)
    proc = subprocess.Popen(
        command, stdout=sys.stdout, stderr=sys.stderr, shell=True,
        universal_newlines=True)
    proc.wait()


def maybe_resolve_path_conflict(path):
    """creates a unique filename incase `path` already exists"""
    base_dir, filename = os.path.split(path)

    sep = filename.index('.')
    name, ext = filename[:sep], filename[sep:]

    while os.path.exists(path):
        if "_" in name:
            name_split = name.split("_")
            i = int(name_split[-1])
            name = name_split[0] + f"_{i + 1}"
        else:
            name += "_1"

        path = os.path.join(base_dir, name + ext)

    return path


def export(data, out_path):
    out_path = maybe_resolve_path_conflict(out_path)
    with open(out_path, 'w') as f:
        f.write("\n".join(data))
    return out_path


def main(args):
    begin = time.time()

    # we will serialize the data so that only your very first run will be
    #  time-consuming. If you change any part of the preprocessing code, you will
    #  need to make sure that `pickle_path` doesn't exist or modify code below
    pickle_path = './data.pickle'
    if not os.path.exists(pickle_path):
        processor = Processor(args.batch_size,
                              enc_max_len=args.enc_max_len,
                              dec_max_len=args.dec_max_len)

        data = processor.preprocess(args.data_dir,
                                    do_reverse=args.do_reverse,
                                    do_lower=args.do_lower)

        print("Serializing data to", pickle_path)
        with open(pickle_path, 'wb') as f:
            pickle.dump((processor, data), f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print("Loading pickled data..")
        with open(pickle_path, 'rb') as f:
            processor, data = pickle.load(f)

    # seq2seq constituency parser
    parser = Parser(processor,
                    emb_dim=args.emb_dim,
                    num_layers=args.num_layers,
                    hidden_dim=args.hidden_dim,
                    batch_size=args.batch_size,
                    attention=args.attention)

    # data[0] and data[1] are train and dev set respectively
    print("Training..")
    parser.train(data[0], data[1], batch_size=args.batch_size, epochs=args.epochs)

    # data[2] is the test data, and data[2][0] is the encoder inputs
    print("Predicting..")
    preds = parser.predict(data[2][0], verbose=args.verbose)

    # data[2][-1] is the decoder outputs, i.e. gold sequence labels, as list
    golds = data[2][-1]

    # we will export `preds` and `golds` before post-processing, in case you want
    # to experiment with different postprocessing methods separately
    print("Exporting golds and preds..")
    gold_path = './golds.out'
    gold_path = export(golds, gold_path)

    pred_path = './preds.out'
    pred_path = export(preds, pred_path)

    print("Postprocessing..")
    preds = postprocess(preds)
    golds = postprocess(golds)

    print("Exporting postprocessed golds and preds..")
    gold_pp_path = gold_path + '.pp'
    gold_pp_path = export(golds, gold_pp_path)

    pred_pp_path = pred_path + '.pp'
    pred_pp_path = export(preds, pred_pp_path)

    # automatic execution of EVALB script
    run_evalb(args.evalb_dir, gold_path=gold_pp_path, pred_path=pred_pp_path)

    print("Execution Time: {:.2f}s".format(time.time() - begin))


if __name__ == '__main__':
    args = argparser.parse_args()

    print("******** FLAGS ********")
    for k, v in vars(args).items():
        print(f"\t{k}: {v}")
    print()

    main(args)
