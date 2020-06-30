#! /usr/bin/python
# -*- coding: utf-8 -*-



"""
Created on Tue March  10 06:10:34 2020
@author: annapustova
"""


import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import Dense, Dropout, Input
from tensorlayer.layers.core import Layer
from tensorlayer.models import Model


class Seq2seqEnsemble(Model):

    def __init__(self, model1, model2, decoder_seq_length, n_units, model2_coef=2, name=None):
        super(Seq2seqEnsemble, self).__init__(name=name)

        self.model1 = model1
        self.model2 = model2
        self.model2_coef = model2_coef

        self.reshape_layer = tl.layers.Reshape([-1, n_units])
        self.dense_layer = tl.layers.Dense(
            n_units=self.model1.vocabulary_size, in_channels=n_units)
        self.reshape_layer_after = tl.layers.Reshape(
            [-1, decoder_seq_length, self.model1.vocabulary_size])
        self.reshape_layer_individual_sequence = tl.layers.Reshape(
            [-1, 1, self.model1.vocabulary_size])

    def inference(self, encoding, seq_length, start_token, top_n):
        """Inference mode"""
        """
        Parameters
        ----------
        encoding : input tensor
            The source sequences
        seq_length : int
            The expected length of your predicted sequence.
        start_token : int
            <SOS> : The token of "start of sequence"
        top_n : int
            Random search algorithm based on the top top_n words sorted by the probablity. 
        """
        feed_output1 = self.model1.embedding_layer(encoding[0])
        feed_output2 = self.model2.embedding_layer(encoding[0])

        state1 = [None for i in range(self.model1.n_layer)]
        state2 = [None for i in range(self.model2.n_layer)]

        for i in range(self.model1.n_layer):
            feed_output1, state1[i] = self.model1.enc_layers[i](
                feed_output1, return_state=True)
        for i in range(self.model2.n_layer):
            feed_output2, state2[i] = self.model2.enc_layers[i](
                feed_output2, return_state=True)

        for i, layer in enumerate(state1):
            for j, _ in enumerate(layer):
                state1[i][j] = tf.math.add(
                    state1[i][j], tf.math.scalar_mul(self.model2_coef, state2[i][j]))
                state1[i][j], _ = tf.linalg.normalize(state1[i][j])
        state = state1

        batch_size = len(encoding[0].numpy())
        decoding = [[start_token] for i in range(batch_size)]
        feed_output = self.model1.embedding_layer(decoding)

        for i in range(self.model1.n_layer):
            feed_output, state[i] = self.model1.dec_layers[i](
                feed_output, initial_state=state[i], return_state=True)

        feed_output = self.reshape_layer(feed_output)
        feed_output = self.dense_layer(feed_output)
        feed_output = self.reshape_layer_individual_sequence(feed_output)
        feed_output = tf.argmax(feed_output, -1)
        # [B, 1]
        final_output = feed_output

        for i in range(seq_length - 1):
            feed_output = self.model1.embedding_layer(feed_output)
            for i in range(self.model1.n_layer):
                feed_output, state[i] = self.model1.dec_layers[i](
                    feed_output, initial_state=state[i], return_state=True)
            feed_output = self.reshape_layer(feed_output)
            feed_output = self.dense_layer(feed_output)
            feed_output = self.reshape_layer_individual_sequence(feed_output)
            ori_feed_output = feed_output
            if (top_n is not None):
                for k in range(batch_size):
                    idx = np.argpartition(
                        ori_feed_output[k][0], -top_n)[-top_n:]
                    probs = [ori_feed_output[k][0][i] for i in idx]
                    probs = probs / np.sum(probs)
                    feed_output = np.random.choice(idx, p=probs)
                    feed_output = tf.convert_to_tensor(
                        [[feed_output]], dtype=tf.int64)
                    if (k == 0):
                        final_output_temp = feed_output
                    else:
                        final_output_temp = tf.concat(
                            [final_output_temp, feed_output], 0)
                feed_output = final_output_temp
            else:
                feed_output = tf.argmax(feed_output, -1)
            final_output = tf.concat([final_output, feed_output], 1)

        return final_output, state

    def forward(self, inputs, seq_length=20, start_token=None, return_state=False, top_n=None):

        encoding = inputs
        output, state = self.inference(
            encoding, seq_length, start_token, top_n)

        if (return_state):
            return output, state
        else:
            return output
