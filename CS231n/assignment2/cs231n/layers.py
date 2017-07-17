import numpy as np

def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
  We multiply this against a weight matrix of shape (D, M) where
  D = \prod_i d_i

  Inputs:
  x - Input data, of shape (N, d_1, ..., d_k)
  w - Weights, of shape (D, M)
  b - Biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  xr = x.reshape(x.shape[0], -1)
  out = np.dot(xr, w) + b
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx = np.dot(dout, w.T).reshape(x.shape)
  dw = np.dot(x.reshape(x.shape[0], -1).T, dout)
  db = np.sum(dout, axis = 0)
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = np.maximum(0, x)
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  x = cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  dx = dout.copy()
  dx[x <= 0] = 0
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """

  N, C, H, W = x.shape
  F, C, HH, WW = w.shape
  pad = conv_param['pad']
  stride = conv_param['stride']
  x1 = np.lib.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), 'constant')
  N1, C1, H1, W1 = x1.shape
  Hp = int(1 + (H + 2 * pad - HH) / stride)
  Wp = int(1 + (W + 2 * pad - WW) / stride)
  x2 = np.zeros((N1, C1*HH*WW, Hp * Wp ))
  w1 = w.reshape(F,-1)
  out1 = np.zeros((N1, F, Hp * Wp))
  b1 = np.dot(b.reshape(F,-1), np.ones((1, Hp * Wp )))
 
  for n in range(N1):
      k = 0
      for i in range(0, H1-HH+1, stride):
          for j in range(0, W1-WW+1, stride):
              x2[n,:, k] = x1[n,:,i:i+HH, j:j+WW].reshape(C1*HH*WW,)
              k += 1
      out1[n,:,:] = np.dot(w1, x2[n,:,:]) + b1
    
  
  out = out1.reshape(N1, F, Hp, Wp)


  
####
  cache = (x, w, b, conv_param)
  return out, cache



def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  x, w, b, conv_param = cache
  p = conv_param['pad']
  s = conv_param['stride']
  dx, dw, db = None, None, None
  dx = np.zeros_like(x)
  N, C, H, W = x.shape
  F, C, HH, WW = w.shape
  
  for n in range(N):
      for c in range(C):
          for h in range(H):
              for g in range(W):
                  delta = 0.0
                  l0 = int(np.ceil((h+p-HH-1)/s))
                  l0 = np.maximum(0, l0)
                  l1 = int(np.floor((h+p)/s))
                  k0 = int(np.maximum(0,np.ceil((g+p-WW-1)/s)))
                  k1 = int(np.floor((g+p)/s))
                  for l in range(l0,l1+1):
                      for k in range(k0, k1+1):
                          for f in range(F):
                              print("h=%d, p=%d, l=%d, s=%d, g=%d, k=%d" % (h,p,l,s, g,k))
                              print("k0=%d" % (k0))
                              a0 = w[f,c,h + p - l * s, g + p - k*s]
                              a1 = dout[n,f,l,k]
                              delta = delta +  a0 * a1
          
                  dx[n,c,h,g] = delta  
#  
#   Input:
#  - x: Input data of shape (N, C, H, W)
#  - w: Filter weights of shape (F, C, HH, WW)
#  - b: Biases, of shape (F,)
#  - conv_param: A dictionary with the following keys:
#    - 'stride': The number of pixels between adjacent receptive fields in the
#      horizontal and vertical directions.
#    - 'pad': The number of pixels that will be used to zero-pad the input.
#
#  Returns a tuple of:
#  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
#    H' = 1 + (H + 2 * pad - HH) / stride
#    W' = 1 + (W + 2 * pad - WW) / stride
#  - cache: (x, w, b, conv_param)
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db




##############################################
from cs231n.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient
x = np.random.randn(4, 3, 5, 5) #5,5
w = np.random.randn(2, 3, 3, 3)
b = np.random.randn(2,)
dout = np.random.randn(4, 2, 5, 5)
conv_param = {'stride': 1, 'pad': 1}
cache = (x,w,b,conv_param)

dx_num = eval_numerical_gradient_array(lambda x: conv_forward_naive(x, w, b, conv_param)[0], x, dout)
dw_num = eval_numerical_gradient_array(lambda w: conv_forward_naive(x, w, b, conv_param)[0], w, dout)
db_num = eval_numerical_gradient_array(lambda b: conv_forward_naive(x, w, b, conv_param)[0], b, dout)
out, cache = conv_forward_naive(x, w, b, conv_param)

dx, dw, db = conv_backward_naive(dout, cache)


np.sum(x[0,0,0:2,0:2] * w[0,0,1:3,1:3]) + b[0]

a = w[0,0,1,1] * dout[0,0,0,0] + w[0,0,1,0] * dout[0,0,0,1]
a = a + w[0,0,0,1] * dout[0,0,1,0] + w[0,0,0,0] * dout[0,0,1,1]
a = a + w[1,0,1,1] * dout[0,1,0,0] + w[1,0,1,0] * dout[0,1,0,1]
a = a + w[1,0,0,1] * dout[0,1,1,0] + w[1,0,0,0] * dout[0,1,1,1]


print(dx_num[0,0,0,0])
print(a)

###########

def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx

