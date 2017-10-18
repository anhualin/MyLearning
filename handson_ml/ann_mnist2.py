# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 12:54:03 2017

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

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
    
    
 
def ann_mnist(hidden_units, lrate=0.01, n_epochs = 40, batch_size=50):
    reset_graph()
    X_valid = mnist.validation.images
    y_valid = mnist.validation.labels.astype('int')
    m = mnist.train.num_examples
    n0 = 28 * 28
    n_batches = m // batch_size
    n_classes = 10
    best_loss = np.inf
    epochs_no_progress = 0
    X = tf.placeholder(tf.float32, shape=(None, n0), name='X')
    y = tf.placeholder(tf.int64, shape=(None), name='y')
    with tf.name_scope('dnn'):
        Z = X
        for i in range(len(hidden_units)):
            Z = tf.layers.dense(Z, hidden_units[i], name='level' + str(i), activation=tf.nn.relu)
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
    saver = tf.train.Saver()
    final_model_path = '/tmp/my_deep_mnist_model'
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            for batch_index in range(n_batches):
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
                accuracy_val, loss_val, accuracy_summary_str, loss_summary_str = sess.run([
                        accuracy, loss, accuracy_summary, loss_summary], feed_dict={X:X_valid, y:y_valid})
            if epoch % 5 == 0:
#                print("Epoch:", epoch,
#                  "\tValidation accuracy: {:.3f}%".format(accuracy_val * 100),
#                  "\tLoss: {:.5f}".format(loss_val)
                print('Epoch:', epoch, 
                      '\t Loss: {:.5f}', loss_val, 
                      '\t Validation_accuracy:', accuracy_val)
                if loss_val < best_loss:
                    best_loss = loss_val
                    saver.save(sess, final_model_path)
                else:
                    epochs_no_progress += 5
                    if epochs_no_progress > 10:
                        print('Early Stopping')
                        break

ann_mnist([300, 150, 75], lrate=0.01, n_epochs = 20, batch_size=400)

# below not working, need to further study saver 
with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, '/tmp/my_deep_mnist_model')
    accuracy_val = accuracy.eval(feed_dict={X: X_test, y:y_test})
    print(accuracy_val)

sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, '/tmp/my_deep_mnist_model')
sess.run(loss, feed_dict={X: X_test, y:y_test})
