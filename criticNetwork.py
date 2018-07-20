#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on July 2018

@author: nicola ancona
"""

import tensorflow as tf
import tflearn

class Critic(object):

    def __init(self):
        pass

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])
        net = tflearn.fully_connected(inputs, 400)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)

        t1 = tflearn.fully_connected(net, 300)
        t2 = tflearn.fully_connected(action, 300)

        net = tflearn.activation(tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + \
                                  t2.b, activation='relu')
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)

        return inputs, action, out