import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  N=X.shape[0]
  D=X.shape[1]
  C=W.shape[1]

  scores=X.dot(W)
  scores-=scores.max(axis = 1).reshape(N, 1)

  es=np.exp(scores)
  ess=np.sum(es,axis=1)
  P=(es.T/ess.T).T
  Sn=(np.ones(N).reshape(N,1)).dot(np.array(range(C)).reshape(1,C))
  yn=y.reshape(N,1)
  Y=(Sn==yn).astype(int)
  #
  for i in range(D):             
        for j in range(C):
            dW[i,j]+=((X[:,i]).T).dot(P[:,j]-Y[:,j])
  #
  for k in range(N):
        loss-=np.log(es[k,y[k]]/ess[k])
  dW/=N
  dW+=2*reg*W
  loss/=N
  loss+=reg*np.sum(W*W)

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
 
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
  N=X.shape[0]
  D=X.shape[1]
  C=W.shape[1]
  scores=X.dot(W)
  es=np.exp(scores)
  ess=np.sum(es,axis=1)
  P=(es.T/ess.T).T

  loss=np.sum(-np.log(P[range(N),y]))

  Sn=(np.ones(N).reshape(N,1)).dot(np.array(range(C)).reshape(1,C))
  yn=y.reshape(N,1)
  Y=(Sn==yn).astype(int)
  dW=(X.T).dot(P-Y)

  dW=dW/N
  dW+=reg*2*W
  loss/=N
  loss+=reg*np.sum(W*W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

