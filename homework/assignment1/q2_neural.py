import numpy as np
import random

from q1_softmax import softmax, softmax_grad
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive

def forward_backward_prop(data, labels, params, dimensions):
    """ 
    Forward and backward propagation for a two-layer sigmoidal network 
    
    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    z1 = data.dot(W1)+b1
    h = sigmoid(z1)
    z2 = h.dot(W2)+b2
    yhat = softmax(z2)
    cost = -np.sum(labels*np.log(yhat))
    ### END YOUR CODE
    
    ### YOUR CODE HERE: backward propagation
    # gradQ holds the gradient of \sum_i y_ilog yhat_i(...)
    N = data.shape[0]
    Dyhat = -labels/yhat
    Dz2 = np.zeros_like(z2)
    for i in xrange(Dz2.shape[0]):
        Dz2[i,:] = Dyhat[i,:].reshape((1,-1)).dot(softmax_grad(z2[i,:].reshape((1,-1))))
    Db2 = np.ones((1,N)).dot(Dz2)
    DW2 = h.T.dot(Dz2)
    Dh = Dz2.dot(W2.T)
    Dz1 = Dh*(sigmoid_grad(z1))
    Db1 = np.ones((1,N)).dot(Dz1)
    DW1 = data.T.dot(Dz1)
    
    ### END YOUR CODE
    
    ### Stack gradients (do not modify)
    grad = np.concatenate((DW1.flatten(), Db1.flatten(), 
        DW2.flatten(), Db2.flatten()))
    
    return cost, grad

def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using 
    gradcheck.
    """
    print "Running sanity check..."

    N = 3
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i,random.randint(0,dimensions[2]-1)] = 1
    
    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )
    gradcheck_naive(lambda params: forward_backward_prop(data, labels, params,
         dimensions), params)


def your_sanity_checks(): 
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py 
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    N = 25
    dimensions = [25,50,32]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i,random.randint(0,dimensions[2]-1)] = 1
    
    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )
    gradcheck_naive(lambda params: forward_backward_prop(data, labels, params,
         dimensions), params)
    ### END YOUR CODE

if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
