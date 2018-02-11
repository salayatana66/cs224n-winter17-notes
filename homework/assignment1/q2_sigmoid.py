import numpy as np
from q2_gradcheck import gradcheck_naive

def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    """
    
    ### YOUR CODE HERE
    x = 1.0/(1.0+np.exp(-x))
    ### END YOUR CODE
    
    return x

def sigmoid_grad(f):
    """
    Compute the gradient for the sigmoid function here. Note that
    for this implementation, the input f should be the sigmoid
    function value of your original input x. 
    """
    
    ### YOUR CODE HERE
    f = sigmoid(f)*(1.0-sigmoid(f))
    ### END YOUR CODE
    
    return f

def test_sigmoid_basic():
    """
    Some simple tests to get you started. 
    Warning: these are not exhaustive.
    """
    print "Running basic tests..."
    x = np.array([[1, 2], [-1, -2]])
    f = sigmoid(x)
    g = sigmoid_grad(x)
    assert np.amax(g - (sigmoid(x+1e-4)-sigmoid(x-1e-4))/(2e-4)) <= 1e-3
    print "You should verify these results!\n"

def test_sigmoid(): 
    """
    Use this space to test your sigmoid implementation by running:
        python q2_sigmoid.py 
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print "Running your tests..."
    ### YOUR CODE HERE
    z1=np.random.randn(20*3)
    assert np.amax(np.abs(sigmoid_grad(z1) -(sigmoid(z1+1e-3)-sigmoid(z1-1e-3))/(2e-3))) <= 1e-3
    ### END YOUR CODE

if __name__ == "__main__":
    test_sigmoid_basic();
    test_sigmoid()
