#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 21:17:24 2017

@author: alin
"""

from sys import platform
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from functools import partial


if platform == 'win32':
    tmpdir = 'C:/Users/alin/Documents/Temp/data'
    modeldir = 'C:/Users/alin/Documents/Temp/model'
else:
    tmpdir = '/tmp/data'
    modeldir = '/tmp/model'
    
    
mnist = input_data.read_data_sets(tmpdir)
X_train = mnist.train.images
X_test = mnist.test.images
y_train = mnist.train.labels.astype('int')
y_test = mnist.test.labels.astype('int')
X_valid = mnist.validation.images
y_valid = mnist.validation.labels.astype('int')

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
    
he_init = tf.contrib.layers.variance_scaling_initializer()

def dnn(inputs, n_hidden_layers=5, n_neurons=100, name=None, activation=tf.nn.elu,
        initializer=he_init):
    with tf.variable_scope('dnn'):
        for layer in range(n_hidden_layers):
            inputs = tf.layers.dense(inputs, n_neurons, activation=activation,
                                     kernel_initializer=initializer,
                                     name='hidden%d' %(layer + 1))
        return inputs