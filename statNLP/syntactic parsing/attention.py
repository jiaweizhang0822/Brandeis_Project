#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""Attention

Feel free to change any part of this code
"""
import tensorflow as tf
from tensorflow.keras.layers import Layer

__author__ = 'Jayeol Chun'


class Attention(Layer):
    """TODO: implement Bahdanau and Luong Attention

  While their mechanism is largely similar, the key difference lies in how they
  compute the score between encoder hidden states and target hidden state.

  https://www.tensorflow.org/tutorials/text/nmt_with_attention#write_the_encoder_and_decoder_model
  """

    def __init__(self, units, mode='bahdanau'):
        super().__init__()

    def call(self, dec_hidden_state, enc_hidden_states):
        pass
