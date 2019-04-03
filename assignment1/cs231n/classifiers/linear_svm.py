import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero
  h=0.01

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    #print(scores)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):    
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j]+=X[i]
        dW[:,y[i]]-=X[i]
        
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W

  """
  for i in range(W.shape[0]):
    for j in range(W.shape[1]):
      Wt = W
      Wt[i,j]+=h        
      loss_t = 0.0
      
      for k in range(num_train):
        scores_t = X[k].dot(Wt)
        correct_class_score_t = scores_t[y[k]]        
        for l in range(num_classes):           
          if l == y[k]:
            continue
          margin_t = scores_t[l] - correct_class_score_t + 1 # note delta = 1
          if margin_t > 0:
            loss_t += margin_t            
      loss_t /= num_train
      loss_t += reg * np.sum(Wt * Wt)
      
      N=X.shape[0]
      C=W.shape[1]
      S=X.dot(Wt)
      Sn=(np.ones(N).reshape(N,1)).dot(np.array(range(C)).reshape(1,C))
      yn=y.reshape(N,1)
      ind=(Sn==yn)
      right=S[ind]
      t=(S.T-right.T+1).T
      t[ind]=0
      t=np.maximum(0,t)
      loss_t=np.sum(t)/(X.shape[0])+reg*np.sum(Wt*Wt)
      
    
      dW[i,j]=loss_t
    dW = (dW-loss)/h
    """


  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  W=np.array(W)
  X=np.array(X)
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################

  N=X.shape[0]
  C=W.shape[1]
  S=X.dot(W)
  Sn=(np.ones(N).reshape(N,1)).dot(np.array(range(C)).reshape(1,C))
  yn=y.reshape(N,1)
  ind=(Sn==yn)
  right=S[ind]
  t=(S.T-right.T).T+1
  t[ind]=0
  t=np.maximum(0,t)
  loss=np.sum(t)/N+reg*np.sum(W*W)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
  R=(t>0).astype(int)
  R[ind]=-np.sum(R,axis=1)
  dW=np.dot(X.T,R)/N+2*reg*W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
