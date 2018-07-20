#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on July 2018

@author: nicola ancona
"""

import tensorflow as tf
import tflearn

class Actor(object):

    def __init__(self):
        pass

    def create_actor_network(self):
        inputs = tflearn.input_data(shape=[None, self.sdim])
        net = tflearn.fully_connected(inputs, 400)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, 300)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)

        # initializig net weights
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)

        out = tflearn.fully_connected(net, self.a_dim, activation='tanh', weights_init=w_init)

        scaled_out = tf.multiply(out, self.action_bound)
        return inputs, out, scaled_out