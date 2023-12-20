import numpy as np

def sigmoid(x,b):
    return (2/(1+np.exp(-b*x)))-1
