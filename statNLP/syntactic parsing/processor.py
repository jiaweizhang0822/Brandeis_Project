#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""Pre-processing & Post-processing

Feel free to change any part of this code
"""
import os

import nltk
import tensorflow as tf
import tqdm

__author__ = 'Jiawei Zhang'


##################################### Util #####################################
def split_corpus(data, sep_a, sep_b):
    """splits into train, dev and test using `sep_a` and `sep_b`

  Args:
    data: array, output of `tokenize`
    sep_a: int, index separating train and dev
    sep_b: int, index separating dev and test

  Returns:
    tuple, (train, dev, test)
  """
    train = data[:sep_a]
    dev = data[sep_a:-sep_b]
    test = data[-sep_b:]
    return train, dev, test


def max_length(data):
    """computes max length of all arrays inside `data`"""
    return max(len(x) for x in data)


def tokenize(data, tokenizer, max_len=-1):
    """tokenizes text and converts each token into index while padding/truncating

  Args:
    data: list of strings
    tokenizer: tf.keras.preprocessing.text.Tokenizer object
    max_len: int, -1 for unspecified max_len

  Returns:
    array of ints, where each int corresponds to an index for a token
  """
    tensor = tokenizer.texts_to_sequences(data)

    # safer to delete from the back, in case user specifies max_len
    kwargs = {'padding': 'post', 'truncating': 'post'}
    if max_len > 0:
        kwargs['maxlen'] = max_len

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, **kwargs)
    return tensor


def fit_tokenizer(data, unk=None, lower=False):
    """fits a tensorflow Tokenizer object

  Args:
    data: list of str
    unk: str, which token to use for oov_token
    lower: bool, whether to lower-case string

  Returns:
    tf.keras.preprocessing.text.Tokenizer object
  """
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        filters='', oov_token=unk, lower=lower)
    tokenizer.fit_on_texts(data)
    return tokenizer


def linearize_parse_tree(tree):
    """TODO: implement a linearization method to turn a tree into a sequence

    As is, we simply keep parenthesis and pos tags while dropping word tokens.
    This is only to ensure that your system (or school server) doesn't run out of
    memory. We don't want the decoder's vocabulary to be unnecessarily big.

    Feel free to modify this or implement your own custom linearization function
    from scratch

    Args:
        tree: nltk.tree.Tree object

    Returns:

    """
    tree = tree.__str__().replace("\n", "").split()

    out = []
    stack = []
    for i, tok in enumerate(tree):
        if '(' in tok:
            out.append(tok)
            stack.append(tok[tok.index('(') + 1:])
        else:
            out[-1] = out[-1][out[-1].index('(') + 1:]
            stack.pop()
            idx = tok.index(")")
            for _ in range(idx + 1, len(tok)):
                out.append(')' + stack.pop())

    return out


#################################### Loader ####################################
def load_dataset(dataset_dir):
    """loads a single dataset (train, dev, test)

  Note that technically this is not loading, which happens when we iterate
    through the CorpusView objects later.

  Args:
    dataset_dir: str, path to root of a single dataset in PTB

  Returns:
    tuple, ConcatenatedCorpusView objects corresponding to raw sentences and
      parse trees
  """
    reader = nltk.corpus.BracketParseCorpusReader(dataset_dir, r'.*/wsj_.*\.mrg')
    sents = reader.sents()
    trees = reader.parsed_sents()
    return sents, trees


def load_data(data_dir):
    """loads Penn TreeBank

  Args:
    data_dir: str, path to root of PTB

  Returns:
    dict, where keys are 'dev', 'train', 'test' and values are return values
      from `load_dataset` above
  """
    data = {}
    datasets = ['dev', 'train', 'test']
    for dataset in datasets:
        dataset_dir = os.path.join(data_dir, dataset)
        data[dataset] = load_dataset(dataset_dir)
    return data


################################# PostProcess ##################################
def postprocess(lin_tree, snt):
    """"

    The purpose of post-processing is to ensure that the model's prediction parse
    trees are well-formed, i.e. the number of opening brackets match the number of
    closing brackets. How you deal with bracket mis-matches is entirely up to you.

    Args:
        lin_tree: str
        snt: list of words consists of sentence

    Returns:
        out

    """

    lin_trees = lin_tree.split()
    snt.extend(['<PAD>' for i in range(500)]) # pad for words in lin_tree > words in snt
    out = ''
    left_paren = 0
    right_paren = 0
    for word in lin_trees:
        left_paren += word.count('(')
        right_paren += word.count(')')

    i = 0 # word pointer
    for word in lin_trees:
        if "(" in word:
            out += word
        elif ")" in word:
            out += ')'
        else:
            out += '(' + word + ' ' + snt[i] + ')'
            i += 1
    '''
    if left_paren > right_paren:
        out+= ')' * (left_paren-right_paren)
    elif left_paren < right_paren:
        out = '(' * (right_paren-left_paren)+ out
    '''
    return out



################################# Preprocessor #################################
class Processor(object):
    def __init__(self, batch_size, enc_max_len=-1, dec_max_len=-1):
        self.batch_size = batch_size
        self.enc_max_len = enc_max_len
        self.dec_max_len = dec_max_len

        # to be instantiated later
        self.enc_tokenizer = None
        self.dec_tokenizer = None

        # consts
        self.END = '<end>'
        self.UNK = '<unk>'

    def preprocess(self, data_dir, do_reverse=False, do_lower=False):
        """loads and preprocesses PTB

    1. loads data with NLTK
    2. performs a preliminary processing on both sentences and parse trees
    3. collects vocab
    4. tokenize and convert strings into indices
    5. group train, dev and test data

    Args:
      data_dir: str, path to root of PTB
      do_reverse: bool, whether to reverse encoder-input sentences
      do_lower: bool, whether to lower-case

    Returns:
      tuple
    """
        if not os.path.exists(data_dir):
            raise ValueError("data_dir doesn't exist in 'preprocess'")

        # 1. loads data
        data = load_data(data_dir)

        # 2. preliminary processing
        for dataset, datum in data.items():
            print("Loading "+dataset )
            sents, trees = datum

            # actual loading of dataset
            _sents = [sent if not do_reverse else list(reversed(sent))
                      for sent in tqdm.tqdm(sents)]

            _trees, _labels, _lin_trees = [], [], []
            for tree in tqdm.tqdm(trees):
                lin_tree = linearize_parse_tree(tree)

                _trees.append([self.END] + lin_tree)
                _labels.append(lin_tree + [self.END])
                _lin_trees.append(" ".join(lin_tree))  # for evaluating test

            # override original data
            data[dataset] = (_sents, _trees, _labels, _lin_trees)

            print("Sample data from", dataset)
            print("\tSent:", _sents[0])
            print("\tTree:", _trees[0])
            print("\tLabel:", _labels[0])

        # 3. collect vocab from train and dev set only
        train_dev_sents = data['train'][0] + data['dev'][0]
        train_dev_trees = data['train'][1] + data['dev'][1]

        self.enc_tokenizer = fit_tokenizer(
            train_dev_sents, unk=self.UNK, lower=do_lower)
        self.dec_tokenizer = fit_tokenizer(train_dev_trees)

        # + 1 for padding, which gets the index 0
        self.enc_vocab_size = len(self.enc_tokenizer.word_index) + 1
        self.dec_vocab_size = len(self.dec_tokenizer.word_index) + 1

        print("\nEncoder Vocab Size:", self.enc_vocab_size)
        print("Decoder Vocab Size:", self.dec_vocab_size)
        print("Decoder Vocabs:", self.dec_tokenizer.word_index)

        # 4. tokenize and convert to indices all together
        all_sents = train_dev_sents + data['test'][0]
        all_trees = train_dev_trees + data['test'][1]
        all_labels = data['train'][2] + data['dev'][2] + data['test'][2]

        enc_inputs = tokenize(all_sents, self.enc_tokenizer, self.enc_max_len)
        dec_inputs = tokenize(all_trees, self.dec_tokenizer, self.dec_max_len)
        dec_outputs = tokenize(all_labels, self.dec_tokenizer, self.dec_max_len)

        print("\nSample transformed data")
        print("\tEncoder Inputs as Indices:", enc_inputs[0])
        print("\tEncoder Inputs as Text: {}".format(
            [self.enc_tokenizer.index_word[x] for x in enc_inputs[0] if x > 0]))
        print("\tDecoder Inputs as Indices:", dec_inputs[0])
        print("\tDecoder Inputs as Text: {}".format(
            [self.dec_tokenizer.index_word[x] for x in dec_inputs[0] if x > 0]))
        print("\tDecoder Targets as Indices:", dec_outputs[0])
        print("\tDecoder Targets as Text: {}".format(
            [self.dec_tokenizer.index_word[x] for x in dec_outputs[0] if x > 0]))

        if self.enc_max_len <= 0:
            self.enc_max_len = max_length(enc_inputs)

        if self.dec_max_len <= 0:
            self.dec_max_len = max_length(dec_inputs)
            assert self.dec_max_len == max_length(dec_outputs)

        print("\nEncoder Input Max Length:", self.enc_max_len)
        print("Decoder Input Max Length:", self.dec_max_len)

        # 5. splits and groups train, dev and test
        self.train_len = len(data['train'][0])
        self.dev_len = len(data['dev'][0])
        self.test_len = len(data['test'][0])

        enc_inputs_train, enc_inputs_dev, enc_inputs_test = split_corpus(
            enc_inputs, self.train_len, self.test_len)
        dec_inputs_train, dec_inputs_dev, _ = split_corpus(
            dec_inputs, self.train_len, self.test_len)
        dec_outputs_train, dec_outputs_dev, _ = split_corpus(
            dec_outputs, self.train_len, self.test_len)

        print("\nTensor Shapes:")
        print("\tTrain: enc inputs {} | dec inputs {} | dec outputs {}".format(
            enc_inputs_train.shape, dec_inputs_train.shape, dec_outputs_train.shape))
        print("\tDev: enc inputs {} | dec inputs tgt {} | dec outputs {}".format(
            enc_inputs_dev.shape, dec_inputs_dev.shape, dec_outputs_dev.shape))
        print("\tTest: enc inputs {}".format(enc_inputs_test.shape))

        train_dataset = [enc_inputs_train, dec_inputs_train, dec_outputs_train]
        dev_dataset = [enc_inputs_dev, dec_inputs_dev, dec_outputs_dev]

        # for test set, we will only use the encoder inputs during decoding. We will
        # also use the linearized tree to evaluate the quality of our model outputs
        test_dataset = [enc_inputs_test, data['test'][-1]]

        return train_dataset, dev_dataset, test_dataset
