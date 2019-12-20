#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""Encoder-Decoder (Seq2Seq) Model for Constituency Parsing

Feel free to change any part of this code
"""
import numpy as np
import tqdm
from tensorflow.keras.layers import Dense, Embedding, Input, LSTM
from tensorflow.keras.models import Model

from attention import Attention

__author__ = 'Jayeol Chun'


def batchify(instances, batch_size=32):
    """splits instances into batches, each of which contains at most batch_size"""
    batches = [instances[i:i + batch_size] if i + batch_size <= len(instances)
               else instances[i:] for i in range(0, len(instances), batch_size)]
    return batches


class Parser(object):
    def __init__(self, processor, emb_dim=64, num_layers=1, hidden_dim=64,
                 batch_size=64, attention=None):
        # processor
        self._p = processor

        self.emb_dim = emb_dim  # dimension of embedding vector
        self.num_layers = num_layers  # number of LSTM layers
        self.hidden_dim = hidden_dim  # dimension of LSTM hidden units
        self.batch_size = batch_size

        self.attention_mode = attention  # str, which attention to use

        self.build()

    def build(self):
        """TODO: incorporate Attention into the parser"""
        self.attn_layer = None

        # 1. placeholder for inputs for encoder and decoder
        enc_inputs = Input(
            shape=(self._p.enc_max_len,), name='Encoder_Input')
        dec_inputs = Input(
            shape=(self._p.dec_max_len,), name='Decoder_Input')

        # 2. encoder embedding
        self.embedding_layer = Embedding(input_dim=self._p.enc_vocab_size,
                                         output_dim=self.emb_dim,
                                         input_length=self._p.enc_max_len,
                                         mask_zero=True,
                                         name='Encoder_Embedding')
        enc_embed = self.embedding_layer(enc_inputs)

        # 3. encoder LSTM
        for i in range(self.num_layers - 1):
            enc_embed = LSTM(self.hidden_dim, return_sequences=True,
                             name=f'Encoder_LSTM_{i + 1}')(enc_embed)

        # when using more than one layer, this will be the final Encoder LSTM
        self.enc_lstm = LSTM(self.hidden_dim, return_state=True,
                             return_sequences=True, name='Encoder_LSTM')
        enc_outputs, enc_state_h, enc_state_c = self.enc_lstm(enc_embed)
        enc_states = [enc_state_h, enc_state_c]

        # 4. decoder embedding
        self.dec_embedding_layer = Embedding(input_dim=self._p.dec_vocab_size,
                                             output_dim=self.emb_dim,
                                             input_length=self._p.dec_max_len,
                                             mask_zero=True,
                                             name='Decoder_Embedding')
        dec_embed = self.dec_embedding_layer(dec_inputs)

        # 5. decoder LSTM
        for i in range(self.num_layers - 1):
            dec_embed = LSTM(self.hidden_dim, return_sequences=True,
                             name=f'Decoder_LSTM_{i + 1}')(dec_embed)

        # when using more than one layer, this will be the final Decoder LSTM
        self.dec_lstm = LSTM(self.hidden_dim, return_state=True,
                             return_sequences=True, name='Decoder_LSTM')
        dec_outputs, dec_state_h, dec_state_c = self.dec_lstm(
            dec_embed, initial_state=enc_states)

        # 6. decoder dense
        self.dense_layer = Dense(self._p.dec_vocab_size, activation='softmax',
                                 name='Decoder_Dense')
        outputs = self.dense_layer(dec_outputs)

        # 7. model definition
        self.model = Model([enc_inputs, dec_inputs], outputs)

        ### Here we prepare for inference, whose mechanism is fundamentally
        ### different from our training model. Whereas the above model feeds the
        ### gold decoder input to the decoder, we assume that we do not have access
        ### to the gold decoder input during inference (which is our target label,
        ### after all). Hence, we will fetch the internal LSTM states of the encoder
        ### and set them to be the hidden states of LSTM of the decoder, and produce
        ### a prediction (a single token) for every single timestep, one at a time.

        # A. encoder model, receiving encoder inputs and returning encoder states
        self.enc_model = Model(enc_inputs, enc_states)

        # B. placeholders for encoder's hidden states, h and c
        dec_state_h_input = Input(shape=(self.hidden_dim,))
        dec_state_c_input = Input(shape=(self.hidden_dim,))
        dec_state_inputs = [dec_state_h_input, dec_state_c_input]

        # C. decoder LSTM on decoder inputs, which will initially contain a single
        #  <end> character during inference. This is consistent with the decoder
        #  input from our preprocessing stage
        dec_lstm_outputs, dec_state_h, dec_state_c = self.dec_lstm(
            dec_embed, initial_state=dec_state_inputs)
        dec_states = [dec_state_h, dec_state_c]

        # D. prediction of probabilities over each label
        dec_outputs_ = self.dense_layer(dec_lstm_outputs)

        # E. inference model definition
        self.dec_model = Model([dec_inputs] + dec_state_inputs,
                               [dec_outputs_] + dec_states)

        ### Entire model configuration
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        self.model.summary()

    def train(self, train_data, dev_data, batch_size=64, epochs=1):
        """trainer

    Note that you shouldn't trust the `accuracy` too much as what really matters
     is how well the model can do inference. We will care more about the Bracket
     F1 score provided by EVALB.

    Args:
      train_data: list, (enc_input, dec_input, dec_output)
      dev_data: list, (enc_input, dec_input, dec_output)
      batch_size: int
      epochs: int
    """
        self.model.fit([train_data[0], train_data[1]], train_data[2],
                       batch_size=batch_size, epochs=epochs,
                       validation_data=([dev_data[0], dev_data[1]], dev_data[2]))

    def decode_batch(self, batch):
        """decodes a batch at a time"""
        end_idx = self._p.dec_tokenizer.word_index[self._p.END]

        states_value = self.enc_model.predict(batch)

        target_seq = np.zeros((len(batch), self._p.dec_max_len))
        target_seq[:, 0] = end_idx  # <end> symbol

        preds = np.zeros_like(target_seq, dtype=np.int32)
        for i in range(self._p.dec_max_len):
            output_toks, h, c = self.dec_model.predict([target_seq] + states_value)

            pred_tok_idx = np.argmax(output_toks[:, -1, :], axis=1)

            preds[:, i] = pred_tok_idx

            target_seq = np.zeros_like(target_seq)
            target_seq[:, 0] = pred_tok_idx

            # Update states
            states_value = [h, c]

        return preds.tolist()

    def predict(self, data, verbose=False):
        """outputs predictions for `data`"""
        preds = []

        batches = batchify(data, self.batch_size)
        for batch in tqdm.tqdm(batches):
            _preds = self.decode_batch(batch)

            if verbose:
                for pred in preds:
                    pred_str = \
                        [self._p.dec_tokenizer.index_word[x] for x in pred if x > 0]
                    print(pred_str)

            preds.extend(_preds)

        # index-to-token mapping, returns a list of str
        preds = self._p.dec_tokenizer.sequences_to_texts(preds)

        # filter out <end> symbols
        no_end_count = 0
        for i, pred in enumerate(preds):
            try:
                end_idx = pred.index(self._p.END)
                pred = pred[:end_idx].strip()
                preds[i] = pred
            except ValueError:
                no_end_count += 1

        if no_end_count:
            # just a warning; may be due to lack of training or simply that our
            # decoder max len is too small for a particular data instance
            print("[!] No End Count: {} / {}".format(no_end_count, len(data)))

        return preds
