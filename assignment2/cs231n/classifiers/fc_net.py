import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


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
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    
    ############################################################################
    # TODO: Initialize the weights and biases of the two-layer net. Weights    #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases using the keys 'W2' and 'b2'.                                 #
    ############################################################################
    self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
    self.params['b1'] = np.zeros(hidden_dim)
    self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b2'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


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
    ############################################################################
    # TODO: Implement the forward pass for the two-layer net, computing the    #
    # class scores for X and storing them in the scores variable.              #
    ############################################################################
    W1 = self.params['W1']
    b1 = self.params['b1']
    W2 = self.params['W2']
    b2 = self.params['b2']
    out, fcache1 = affine_relu_forward(X, W1, b1)
    scores, fcache2 = affine_forward(out, W2, b2)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the two-layer net. Store the loss  #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss using softmax, and make sure that grads[k] holds the gradients for  #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    loss, dx = softmax_loss(scores, y)
    
    loss += 0.5 * self.reg*(np.sum(W1 * W1) + np.sum(W2 * W2))
    
    dx, dW2, db2 = affine_backward(dx, fcache2)
    dx, dW1, db1 = affine_relu_backward(dx, fcache1)
    
    dW2 += self.reg * W2
    dW1 += self.reg * W1
    
    grads = {'W1': dW1, 'W2': dW2, 'b1': db1, 'b2': db2}
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################
    dims = [input_dim] + hidden_dims + [num_classes]
    self.L = len(dims) - 1
    
    Ws = {'W' + str(i + 1): weight_scale * np.random.randn(dims[i], dims[i + 1])
          for i in range(self.L)}
    bs = {'b' + str(i + 1): np.zeros(dims[i + 1]) for i in range(self.L)}
    
    if use_batchnorm:
        gammas = {'gamma' + str(i + 1): np.ones(hidden_dims[i]) 
                  for i in range(self.L - 1)}
        betas = {'beta' + str(i + 1): np.zeros(hidden_dims[i]) 
                 for i in range(self.L - 1)}
        self.params.update(gammas)
        self.params.update(betas)
    
    self.params.update(Ws)
    self.params.update(bs)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################
    
    # Store the result of each layer in the following dictionary.
    hidden = {}
    hidden['h0'] = X
    
    if self.use_dropout:
        hdrop, cache_hdrop = dropout_forward(hidden['h0'], self.dropout_param)
        hidden['hdrop0'], hidden['cache_hdrop0'] = hdrop, cache_hdrop
    
    for i in range(self.L):
        count = i + 1
        w = self.params['W' + str(count)]
        b = self.params['b' + str(count)]
        if self.use_dropout:
            h = hidden['hdrop' + str(i)]
        else:
            h = hidden['h' + str(i)]
        if count == self.L:
            hout, cache_h = affine_forward(h, w, b)
            hidden['cache_h' + str(count)] = cache_h
        else:
            if not self.use_batchnorm:
                hout, cache_h = affine_relu_forward(h, w, b)
                hidden['cache_h' + str(count)] = cache_h
            else:
                hout, cache_h = affine_BatchNormalization_relu_forward(h, w, b, 
                                                                       self.params['gamma' + str(count)],
                                                                       self.params['beta' + str(count)],
                                                                       self.bn_params[i])
                hidden['bn_cache_h' + str(count)] = cache_h
        hidden['h' + str(count)] = hout
        if self.use_dropout:
            h = hidden['h' + str(count)]
            hdrop, cache_hdrop = dropout_forward(h, self.dropout_param)
            hidden['hdrop' + str(count)], hidden['cache_hdrop' + str(count)] = hdrop, cache_hdrop
            
    scores = hidden['h' + str(self.L)]
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    
    # Compute data loss using softmax. Note that we can directly use softmax(x,y)
    # from cs231n.layers. 
    loss, dscores = softmax_loss(scores, y)
    for w in [self.params[f] for f in self.params.keys() if f[0] == 'W']:
        loss += 0.5 * self.reg * np.sum(w * w)
    
    # Now we are going to compute gradients for parameters in each layer.
    hidden['dh' + str(self.L)] = dscores
    for i in range(self.L)[::-1]:
        count = i + 1
        dh = hidden['dh' + str(count)]
        if count == self.L: 
            cache_h = hidden['cache_h' + str(count)]
            dh, dW, db = affine_backward(dh, cache_h)
        else:
            if not self.use_batchnorm:
                cache_h = hidden['cache_h' + str(count)]
                dh, dW, db = affine_relu_backward(dh, cache_h)
            else:
                bn_cache_h = hidden['bn_cache_h' + str(count)]
                dh, dW, db, dgamma, dbeta = affine_BatchNormalization_relu_backward(dh, bn_cache_h)
                grads['beta' + str(count)] = dbeta
                grads['gamma' + str(count)] = dgamma
        hidden['dh' + str(count - 1)] = dh
        hidden['dW' + str(count)] = dW
        hidden['db' + str(count)] = db
    
    list_dw = {key[1:] : val + self.reg * self.params[key[1:]] for key, val in
              hidden.iteritems() if key[:2] == 'dW'}
    
    list_db = {key[1:] : val for key, val in hidden.iteritems() 
               if key[:2] == 'db'}
    
    grads.update(list_dw)
    grads.update(list_db)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads

def affine_BatchNormalization_relu_forward(x, w, b, gamma, beta, bn_param):
  """
  A convenience layer that performs an affine transform followed by a batch
  normalization and then followed by relu.

  Inputs:
  - x: Input to the affine layer of shape (N, D)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape(M,)
  - gamma: Scale parameter fot the batch normalization of shape (D,)
  - beta: Shift parameter for the batch normalization of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
  
  Returns:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  affine_out, fc_cache = affine_forward(x, w, b)
  bn_out, bn_cache = batchnorm_forward(affine_out, gamma, beta, bn_param)
  relu_out, relu_cache = relu_forward(bn_out)
  cache = (fc_cache, bn_cache, relu_cache)
  return relu_out, cache

def affine_BatchNormalization_relu_backward(dout, cache):
  """
  Backward pass for the affine-batch normalization-relu convenience layer
  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout
  
  Returns:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  - dgamma: Gradient with respect to gamma
  - dbeta: Gradient with respect to beta
  """
  fc_cache, bn_cache, relu_cache = cache
  drelu_out = relu_backward(dout, relu_cache)
  dbn_out, dgamma, dbeta = batchnorm_backward(drelu_out, bn_cache)
  dx, dw, db = affine_backward(dbn_out, fc_cache)
  return dx, dw, db, dgamma, dbeta
