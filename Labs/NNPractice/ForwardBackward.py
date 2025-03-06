


import numpy as np
import pandas as pd

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def mean_square_error(y,yi):
    return np.sum(np.square(y - yi)) / (yi.shape[0])



input = np.array([[5,6,7,6,9], [5,6,7,-1,9], [5,6,4,8,9], [-1,1,-8,8,9], [5,6,-9,-9,9]]) / 100
output = np.array([[0,1,1,1]])


##3 
first_layer = np.random.randint(1, 100, size=(5,3)) / 100
first_bias = np.random.randint(1,10, size = (1,3)) / 10

second_layer = np.random.randint(1, 100, size=(3,2)) / 100
second_bias = np.random.randint(1,10, size = (1,2)) / 10

outputlayer = np.random.randint(1, 100, size=(2,1)) / 100
output_bias = np.random.rand() / 10

for i in range(100):
    first_layer_out = relu((input @ first_layer) + first_bias) 
    second_layer_out = relu((first_layer_out @ second_layer) + second_bias) 
    output = softmax((second_layer_out @ outputlayer) + output_bias)

    print(mean_square_error(input,output))


