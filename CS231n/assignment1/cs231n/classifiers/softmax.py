import numpy as np
from random import shuffle

#C = 3
#D = 1
#N = 1
#W0 = np.random.rand(C,D) - 0.5
#X0 = np.random.rand(D,N)
#y0 = [0]
#f0 = W0.dot(X0)
#f1 = f0 - np.max(f0)
#p0 = np.exp(f1)/np.sum(np.exp(f1))
#
#-np.log(np.exp(f0[0,0])/(np.exp(f0[0,0]) + np.exp(f0[1,0]) + np.exp(f0[2,0])))
#def L(W):
#    loss = 0.0
#    for i in range(N):
#      f = W.dot(X0[:, i])
#      f -= np.max(f)
#      p = np.exp(f)/np.sum(np.exp(f))
#      loss += -np.log(p[y0[i]])
#    loss /= N
#    return loss
#
#df1 = eval_numerical_gradient(L, W0)
#l2, df2 = softmax_loss_naive(W0, X0, y0, 0)
#
#def eval_numerical_gradient(f, x):
#  """ 
#  a naive implementation of numerical gradient of f at x 
#  - f should be a function that takes a single argument
#  - x is the point (numpy array) to evaluate the gradient at
#  """ 
#
#  fx = f(x) # evaluate function value at original point
#  grad = np.zeros(x.shape)
#  h = 0.000001
#
#  # iterate over all indexes in x
#  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
#  while not it.finished:
#
#    # evaluate function at x+h
#    ix = it.multi_index
#    old_value = x[ix]
#    x[ix] = old_value + h # increment by h
#    fxh = f(x) # evalute f(x + h)
#    x[ix] = old_value # restore to previous value (very important!)
#
#    # compute the partial derivative
#    grad[ix] = (fxh - fx) / h # the slope
#    it.iternext() # step to next dimension
#
#  return grad


def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[1]
  num_classes = W.shape[0]
  for i in range(num_train):
      f = W.dot(X[:, i])
      f -= np.max(f)
      p = np.exp(f)/np.sum(np.exp(f))
      loss += -np.log(p[y[i]])
      for j in range(num_classes):
          if j == y[i]:
              
              dW[j,:] += (p[j] - 1) * X[:,i].T
          else:
              dW[j,:] += p[j] * X[:,i].T
        
          
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW = dW / num_train  + reg * W

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
