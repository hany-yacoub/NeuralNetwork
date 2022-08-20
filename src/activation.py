import numpy as np

def tanh(x):
    # hyperbolic tangent activation function for hidden layer
        
    f_x = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    return f_x

def softmax(x):

    # softmax for classification layer
     
    ex = np.exp(x)
    sum = np.sum(ex, axis=1, keepdims=True)
    fx = ex / sum 

    return fx