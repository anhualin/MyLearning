#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 15:02:55 2017

@author: alin
"""
from sys import platform
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

if platform == 'win32':
    tmpdir = 'C:/Users/alin/Documents/Temp/data'
else:
    tmpdir = '/tmp/data/'
    
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
    
def leaky_relu(z, name=None):
    return tf.maximum(0.01 * z, z, name=name)

def selu(z, scale=1.0507009873554804934193349852946,
         alpha=1.6732632423543772848170429916717, name=None):
    return scale * tf.where(z >= 0.0, z, alpha * tf.nn.elu(z), name=name)

def ann_mnist(hidden_units, lrate=0.01, n_epochs = 40, batch_size=50):
    reset_graph()
    m = mnist.train.num_examples
    n0 = 28 * 28
    n_batches = m // batch_size
    n_classes = 10
    best_loss = np.inf
    epochs_no_progress = 0
    X = tf.placeholder(tf.float32, shape=(None, n0), name='X')
    y = tf.placeholder(tf.int64, shape=(None), name='y')
    training = tf.placeholder_with_default(False, shape=(), name='training')
    he_init = tf.contrib.layers.variance_scaling_initializer()
    with tf.name_scope('dnn'):
        Z = X
        for i in range(len(hidden_units)):
            Z_pre = tf.layers.dense(Z, hidden_units[i], name='level_pre' + str(i), 
                                kernel_initializer=he_init)
            bn = tf.layers.batch_normalization(inputs=Z_pre, name='bn' + str(i),
                                               training=training, momentum=0.9)
            Z = selu(bn, name='level' + str(i))
        logits = tf.layers.dense(Z, n_classes, name='output')

    with tf.name_scope('loss'):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name='loss')
        loss_summary = tf.summary.scalar('log_loss', loss)
        
    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lrate)
        training_op = optimizer.minimize(loss)
    
    with tf.name_scope('eval'):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        accuracy_summary = tf.summary.scalar('accuracy', accuracy)
        
    init = tf.global_variables_initializer()
    #saver = tf.train.Saver()
    #final_model_path = '/tmp/my_deep_mnist_model'
    means = X_train.mean(axis=0, keepdims=True)
    stds = X_train.std(axis=0, keepdims=True) + 1e-10
    X_valid_scaled = (X_valid - means) / stds
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            for batch_index in range(n_batches):
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                X_batch_scaled = (X_batch -  means) / stds
                sess.run([training_op, extra_update_ops], feed_dict={training: True, X: X_batch_scaled, y: y_batch})
                accuracy_val, loss_val, accuracy_summary_str, loss_summary_str = sess.run([
                        accuracy, loss, accuracy_summary, loss_summary], feed_dict={X:X_valid_scaled, y:y_valid})
            if epoch % 5 == 4:
#                print("Epoch:", epoch,
#                  "\tValidation accuracy: {:.3f}%".format(accuracy_val * 100),
#                  "\tLoss: {:.5f}".format(loss_val)
                print('Epoch:', epoch, 
                      '\t Loss: {:.5f}', loss_val, 
                      '\t Validation_accuracy:', accuracy_val)
                if loss_val < best_loss:
                    best_loss = loss_val
     #               saver.save(sess, final_model_path)
                else:
                    epochs_no_progress += 5
                    if epochs_no_progress > 10:
                        print('Early Stopping')
                        break

ann_mnist([300, 100], lrate=0.01, n_epochs = 10, batch_size=400)


#######################################################
########################################################
reset_graph()

n_inputs = 28 * 28  # MNIST
n_hidden1 = 300
n_hidden2 = 50
n_hidden3 = 50
n_hidden4 = 50
n_hidden5 = 50
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")



with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden1")
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name="hidden2")
    hidden3 = tf.layers.dense(hidden2, n_hidden3, activation=tf.nn.relu, name="hidden3")
    hidden4 = tf.layers.dense(hidden3, n_hidden4, activation=tf.nn.relu, name="hidden4")
    hidden5 = tf.layers.dense(hidden4, n_hidden5, activation=tf.nn.relu, name="hidden5")
    logits = tf.layers.dense(hidden5, n_outputs, name="outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")
    
learning_rate = 0.01

threshold = 1.0

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
grads_and_vars = optimizer.compute_gradients(loss)
capped_gvs = [(tf.clip_by_value(grad, -threshold, threshold), var)
              for grad, var in grads_and_vars]
training_op = optimizer.apply_gradients(capped_gvs)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 20
batch_size = 200

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: X_test,
                                                y: y_test})
        print(epoch, "Test accuracy:", accuracy_val)

    save_path = saver.save(sess, "/tmp/my_model_final.ckpt")