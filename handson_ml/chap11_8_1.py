# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 10:29:42 2017

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

X_train1 = X_train[y_train < 5]
y_train1 = y_train[y_train < 5]
X_test1 = X_test[y_test < 5]
y_test1 = y_test[y_test < 5]
X_valid1 = X_valid[y_valid < 5]
y_valid1 = y_valid[y_valid < 5]

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
    
n_inputs = 28 * 28
n_outputs = 5
learning_rate = 0.005
reset_graph()

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int64, shape=(None), name='y')

dnn_outputs = dnn(X)
logits = tf.layers.dense(dnn_outputs, n_outputs, kernel_initializer=he_init, name='logits')

with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")
        

with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss, name='training_op')
    
with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 100
batch_size = 50
max_checks_without_success = 20
checks_without_success = 0
best_loss = np.inf

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        rnd_idx = np.random.permutation(X_train1.shape[0])
        for rnd_indicies in np.array_split(rnd_idx, len(X_train1) // batch_size):
            X_batch, y_batch = X_train1[rnd_indicies], y_train1[rnd_indicies]
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        loss_val, acc_val = sess.run([loss, accuracy], feed_dict={X: X_valid1, y: y_valid1})
        loss_train, acc_train = sess.run([loss, accuracy], feed_dict={X: X_train1, y: y_train1})
        if loss_val < best_loss:
            best_loss = loss_val
            save_path = saver.save(sess, modeldir + '/my_mnist_model_0_4.ckpt')
            checks_without_success = 0
        else:
            checks_without_success += 1
            if checks_without_success > max_checks_without_success:
                print('Early stopping')
                break
        print('Epoch:', epoch, '  Train loss:', loss_train, '  Valid loss:', loss_val)
        print('Epoch:', epoch, '  Train acc:', acc_train, '   Valid acc:', acc_val)

with tf.Session() as sess:
    saver.restore(sess, modeldir + '/my_mnist_model_0_4.ckpt')
    acc_test = sess.run(accuracy, feed_dict={X: X_test1, y: y_test1})
    print('Test accuracy: ', acc_test)
    
#######################################################3
######################################################
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError

class DNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_hidden_layers=5, n_neurons=100, 
                 optimizer_class=tf.train.AdamOptimizer,
                 learning_rate=0.005, batch_size=40, activation=tf.nn.elu,
                 initializer=he_init, batch_norm_momentum=None,
                 dropout_rate=None, random_state=None):
        self.n_hidden_layers = n_hidden_layers
        self.n_neurons = n_neurons
        self.optimizer_class = optimizer_class
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.activation = activation
        self.initializer = initializer
        self.batch_norm_momentum = batch_norm_momentum
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        self._session = None
    def _dnn(self, inputs):
        for layer in range(self.n_hidden_layers):
            if self.dropout_rate:
                inputs = tf.layers.dropout(inputs, self.dropout_rate, training=self._training)
            inputs = tf.layers.dense(inputs, self.n_neurons, 
                                     kernel_initializer=self.initializer,
                                     name='hidden%d' % (layer + 1))
            if self.batch_norm_momentum:
                inputs = tf.layers.batch_normalization(inputs, momentum=self.batch_norm_momentum,
                                                       training=self._training)
            inputs = self.activation(inputs, name='hidden%d_out' % (layer + 1))
        return inputs
    
    def _build_graph(self, n_inputs, n_outputs):
        if self.random_state is not None:
            tf.set_random_seed(self.random_state)
            np.random.seed(self.random_state)
            
        X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
        y = tf.placeholder(tf.int32, shape=(None), name='y')
        
        if self.batch_norm_momentum or self.dropout_rate:
            self._training = tf.placeholder_with_default(False, shape=(), name='training')
        else:
            self._training = None
        
        dnn_outputs = self._dnn(X)
        logits = tf.layers.dense(dnn_outputs, n_outputs, kernel_initializer=self.initializer,
                                 name='logits')
        Y_proba = tf.nn.softmax(logits, name='Y_proba')
        
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name='loss')
        
        optimizer = self.optimizer_class(learning_rate=self.learning_rate)
        training_op = optimizer.minimize(loss)
        
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')
        
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        
        self._X, self._y = X, y
        self._Y_proba, self._loss = Y_proba, loss
        self._training_op, self._accuracy = training_op, accuracy
        self._init, self._saver = init, saver
    
    def close_session(self):
        if self._session:
            self._session.close()
            
    def _get_model_params(self):
        with self._graph.as_default():
            gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        return {gvar.op.name: value for gvar, value in zip(gvars, self._session.run(gvars))}

    def _restore_model_params(self, model_params):
        gvar_names = list(model_params.keys)
        assign_ops = {gvar_name: self._graph.get_operation_by_name(gvar_name, +'/Assign') for gvar_name in gvar_names}
        init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
        feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
        self._session.run(assign_ops, feed_dict=feed_dict)
        
    def fit(self, X, y, n_epochs=30, X_valid=None, y_valid=None):
        self.close_session()
        
        n_inputs = X.shape[1]
        self.classes_ = np.unique(y)
        n_outputs = len(self.classes_)
        
        self.class_to_index_ = {label: index for index, label in enumerate(self.classes_)}
        
        y = np.array([self.class_to_index_[label] for label in y], dtype=np.int32)
        
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._build_graph(n_inputs, n_outputs)
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            
        max_checks_without_progress = 20
        checks_without_progress = 0
        best_loss = np.inf
        best_params = None
        
        self._session = tf.Session(graph=self._graph)
        with self._session.as_default() as sess:
            self._init.run()
            for epoch in range(n_epochs):
                rnd_idx = np.random.permutation(len(X))
                for rnd_indices in np.array_split(rnd_idx, len(X) // self.batch_size):
                    X_batch, y_batch = X[rnd_indices], y[rnd_indices]
                    feed_dict = {self._X: X_batch, self._y: y_batch}
                    if self._training is not None:
                        feed_dict['training'] = True
                    sess.run(self._training_op, feed_dict=feed_dict)
                    if extra_update_ops:
                        sess.run(extra_update_ops, feed_dict=feed_dick)
                     
                if X_valid is not None and y_valid is not None:
                    loss_val, acc_val = sess.run([self._loss, self._accuracy],
                                                 feed_dict={self._X:X_valid, self._y:y_valid})
                    if loss_val < best_loss:
                        best_params = self._get_model_params()
                        best_loss = loss_val
                        checks_without_progress = 0
                    else:
                        checks_without_progress += 1
                        print("{}\tValidation loss: {:.6f}\tBest loss: {:.6f}\tAccuracy: {:.2f}%".format(
                            epoch, loss_val, best_loss, acc_val * 100))
                        if checks_without_progress > max_checks_without_progress:
                            print("Early stopping!")
                            break
                else:
                    loss_train, acc_train = sess.run([self._loss, self._accuracy],
                                             feed_dict={self._X: X_batch, self._y: y_batch})
                    print("{}\tLast training batch loss: {:.6f}\tAccuracy: {:.2f}%".format(
                        epoch, loss_train, acc_train * 100))
                    
            if best_params:
                self._restore_model_params(best_params)
            return self
        
    def predict_proba(self, X):
        if not self._session:
            raise NotFittedError("This %s instance is not fitted yet" % self.__class__.__name__)
        with self._session.as_default() as sess:
            return self._Y_proba.eval(feed_dict = {self._X: X})
        
    def predict(self, X):
        class_indices = np.argmax(self.predict_proba(X), axis=1)
        return np.array([[self.classes_[class_index]]
                             for class_index in class_indices], np.int32)
    def save(self, path):
        self._saver.save(self._session, path)

    
###############################################################
dnn_clf = DNNClassifier(random_state=42)
dnn_clf.fit(X_train1, y_train1, X_valid=X_valid1, y_valid=y_valid1)
