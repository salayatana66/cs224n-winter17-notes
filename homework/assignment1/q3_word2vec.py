import numpy as np
import random

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad

from collections import Counter
def normalizeRows(x):
    """ Row normalization function """
    # Implement a function that normalizes each row of a matrix to have unit length
    
    ### YOUR CODE HERE
    rowSizes = np.apply_along_axis(np.linalg.norm,axis=1,arr=x)
    
    ### END YOUR CODE
    
    return x/rowSizes.reshape(-1,1)

def test_normalize_rows():
    print "Testing normalizeRows..."
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]])) 
    # the result should be [[0.6, 0.8], [0.4472, 0.8944]]
    print x
    assert (x.all() == np.array([[0.6, 0.8], [0.4472, 0.8944]]).all())
    print ""

def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models """
    
    # Implement the cost and gradients for one predicted word vector  
    # and one target word vector as a building block for word2vec     
    # models, assuming the softmax prediction function and cross      
    # entropy loss.                                                   
    
    # Inputs:                                                         
    # - predicted: numpy ndarray, predicted word vector (\hat{v} in 
    #   the written component or \hat{r} in an earlier version)
    # - target: integer, the index of the target word               
    # - outputVectors: "output" vectors (as rows) for all tokens     
    # - dataset: needed for negative sampling, unused here.         
    
    # Outputs:                                                        
    # - cost: cross entropy cost for the softmax word prediction    
    # - gradPred: the gradient with respect to the predicted word   
    #        vector                                                
    # - grad: the gradient with respect to all the other word        
    #        vectors                                               
    
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!                                                  
    
    ### YOUR CODE HERE
    predicted = predicted.reshape(-1,1)
    cost = -outputVectors[target,:].reshape(1,-1).dot(predicted)
    expV = np.exp(outputVectors.dot(predicted))
    expS = np.sum(expV)
    cost += np.log(expS)
    gradPred = -outputVectors[target,:].reshape(1,-1) + expV.T.dot(outputVectors)/expS
    grad = expV.dot(predicted.reshape(1,-1))/expS
    grad[target,:] -= predicted.reshape(-1)
    ### END YOUR CODE
    
    return cost, gradPred, grad

def negSamplingCostAndGradient(predicted, target, outputVectors, dataset, 
    K=10):
    """ Negative sampling cost function for word2vec models """

    # Implement the cost and gradients for one predicted word vector  
    # and one target word vector as a building block for word2vec     
    # models, using the negative sampling technique. K is the sample  
    # size. You might want to use dataset.sampleTokenIdx() to sample  
    # a random word index. 
    # 
    # Note: See test_word2vec below for dataset's initialization.
    #                                       
    # Input/Output Specifications: same as softmaxCostAndGradient     
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!
    
    ### YOUR CODE HERE
    predicted = predicted.reshape(-1,1)
    negIndices = np.sort(np.array([dataset.sampleTokenIdx() for l in xrange(K)]))
    nUvc = -outputVectors[negIndices,:].dot(predicted)
    snUvc = sigmoid(nUvc)
    gsnUvc = sigmoid_grad(nUvc)
    Uvc = outputVectors[target,:].dot(predicted)
    sUvc = sigmoid(Uvc)
    gsUvc = sigmoid_grad(Uvc)
    cost = -np.log(sUvc)-np.sum(np.log(snUvc))
    gradPred = -gsUvc/sUvc*outputVectors[target,:].reshape(1,-1)
    gradPred += (gsnUvc/snUvc).T.dot(outputVectors[negIndices,:])
    grad = np.zeros_like(outputVectors)
    grad[target,:] = -gsUvc/sUvc * predicted.reshape(-1)
    for i in negIndices:
        prod = -outputVectors[i,:].dot(predicted)
        grad[i,:] += (sigmoid_grad(prod)/sigmoid(prod)).dot(predicted.reshape(1,-1))
    ### END YOUR CODE
    
    return cost, gradPred, grad


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors, 
    dataset, word2vecCostAndGradient = softmaxCostAndGradient):
    """ Skip-gram model in word2vec """

    # Implement the skip-gram model in this function.

    # Inputs:                                                         
    # - currrentWord: a string of the current center word           
    # - C: integer, context size                                    
    # - contextWords: list of no more than 2*C strings, the context words                                               
    # - tokens: a dictionary that maps words to their indices in    
    #      the word vector list                                
    # - inputVectors: "input" word vectors (as rows) for all tokens           
    # - outputVectors: "output" word vectors (as rows) for all tokens         
    # - word2vecCostAndGradient: the cost and gradient function for 
    #      a prediction vector given the target word vectors,  
    #      could be one of the two cost functions you          
    #      implemented above

    # Outputs:                                                        
    # - cost: the cost function value for the skip-gram model       
    # - grad: the gradient with respect to the word vectors         
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!

    ### YOUR CODE HERE
    cost = 0
    gradIn = np.zeros_like(inputVectors)
    gradOut = np.zeros_like(outputVectors)
    #rndstate = random.getstate()
    for w in contextWords:
        #random.setstate(rndstate)  
        _cost, _gradPred, _grad = word2vecCostAndGradient(inputVectors[
            tokens[currentWord],:],
                                                          tokens[w], 
                                                           outputVectors,
                                                          dataset)
        cost += np.sum(_cost)
        gradIn[tokens[currentWord],:] += _gradPred.reshape(-1)
        gradOut += _grad
    ### END YOUR CODE
    
    return cost, gradIn, gradOut

def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors, 
    dataset, word2vecCostAndGradient = softmaxCostAndGradient):
    """ CBOW model in word2vec """

    # Implement the continuous bag-of-words model in this function.            
    # Input/Output specifications: same as the skip-gram model        
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!

    #################################################################
    # IMPLEMENTING CBOW IS EXTRA CREDIT, DERIVATIONS IN THE WRIITEN #
    # ASSIGNMENT ARE NOT!                                           #  
    #################################################################
    
    cost = 0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    predicted = np.zeros(inputVectors.shape[1])
    for w in contextWords:
        predicted += inputVectors[tokens[w],:].reshape(-1)
    _cost, _gradPred, _grad = word2vecCostAndGradient(predicted,
                                                          tokens[currentWord],
                                                          outputVectors,
                                                          dataset)
    cost += _cost
    for w in contextWords:
        gradIn[tokens[w],:] += _gradPred.reshape(-1)
    gradOut += _grad
    ### END YOUR CODE
    
    return cost, gradIn, gradOut

#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C, word2vecCostAndGradient = softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2,:]
    outputVectors = wordVectors[N/2:,:]
    for i in xrange(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)
        
        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1
        
        c, gin, gout = word2vecModel(centerword, C1, context, tokens, inputVectors, outputVectors, dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N/2, :] += gin / batchsize / denom
        grad[N/2:, :] += gout / batchsize / denom
        
    return cost, grad

def test_word2vec():
    # Interface to the dataset for negative sampling
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)
    
    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], [tokens[random.randint(0,4)] \
           for i in xrange(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(731415)
    np.random.seed(29265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print "=== Gradient check for softmax === "
    Input = np.random.randn(10*10)
    def sfxChecker(Input,dims,target):
        offset = 0
        predicted = Input[offset:dims[0]].reshape(-1,1)
        offset += dims[0]
        outputVectors = Input[offset:].reshape(-1,dims[0])
        _cost, _gradPredicted, _grad = softmaxCostAndGradient(predicted, target, outputVectors, dataset)
        return _cost, np.concatenate((_gradPredicted.flatten(),_grad.flatten()))
    
    gradcheck_naive(lambda Input: sfxChecker(Input,(10,4),3),Input)
    print "=== Gradient check for negative sampling ==="
    Input = np.random.randn(20*30)
    def sfxChecker(Input,dims,target):
        offset = 0
        predicted = Input[offset:dims[0]].reshape(-1,1)
        offset += dims[0]
        outputVectors = Input[offset:].reshape(-1,dims[0])
        _cost, _gradPredicted, _grad = negSamplingCostAndGradient(predicted, target, outputVectors, dataset,K=20)
        return _cost, np.concatenate((_gradPredicted.flatten(),_grad.flatten()))
    gradcheck_naive(lambda Input: sfxChecker(Input,(30,4),3),Input)
    Input = np.random.randn(20*30)
    def sfxChecker(Input,dims,target):
        offset = 0
        predicted = Input[offset:dims[0]].reshape(-1,1)
        offset += dims[0]
        outputVectors = Input[offset:].reshape(-1,dims[0])
        _cost, _gradPredicted, _grad = negSamplingCostAndGradient(predicted, target, outputVectors, dataset,K=20)
        return _cost, np.concatenate((_gradPredicted.flatten(),_grad.flatten()))
    gradcheck_naive(lambda Input: sfxChecker(Input,(30,4),2),Input)
    print "==== Gradient check for skip-gram ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5), dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)
    print "\n==== Gradient check for CBOW      ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5), dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)

    print "\n=== Results ==="
    print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print skipgram("c", 1, ["a", "b"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, negSamplingCostAndGradient)
    print cbow("a", 2, ["a", "b", "c", "a"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print cbow("a", 2, ["a", "b", "a", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, negSamplingCostAndGradient)

if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()
