from builtins import range
from builtins import object
from json import load
from matplotlib.pyplot import cla
import numpy as np
import sys

from layers import *
from layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        params = None
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        - params: The parameters of the net, if the input is an empty dictionary, the nets will initialize the parameters using gaussian distribution.
        """

        self.params = params
        self.reg = reg


        if not params: 
          # print("Use random!")
          self.params = {}
          self.params['W1'] = np.random.normal(0.0, weight_scale, (input_dim, hidden_dim))
          self.params['b1'] = np.zeros(hidden_dim)
          self.params['W2'] = np.random.normal(0.0, weight_scale, (hidden_dim, num_classes))
          self.params['b2'] = np.zeros(num_classes)
          
        
    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None

        tempscores, cache1 = affine_relu_forward(X, self.params['W1'], self.params['b1'])
        scores, cache2 = affine_forward(tempscores, self.params['W2'], self.params['b2'])

        """
        tempscores, cache1 = affine_forward(X, self.params['W1'], self.params['b1'])
        tempscores, cache2 = relu_forward(tempscores)
        scores, cache3 = affine_forward(tempscores, self.params['W2'], self.params['b2'])
        """

        if y is None:
            return scores

        loss, grads = 0, {}

        loss, dx = softmax_loss(scores, y)
        if self.reg != 0.0:
          loss += 0.5 * self.reg * np.sum(self.params['W2'] * self.params['W2']) + \
            0.5 * self.reg * np.sum(self.params['W1'] * self.params['W1'])
        dx, grads['W2'], grads['b2'] = affine_backward(dx, cache2)
        dx, grads['W1'], grads['b1'] = affine_relu_backward(dx, cache1)  

        if self.reg != 0.0:
          grads['W1'] += self.reg * self.params['W1']
          grads['W2'] += self.reg * self.params['W2']
        return loss, grads

