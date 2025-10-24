import os
import torch
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        '''Do not call model.forward() directly!'''
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

bit = "[1 0 0 0 1 0 0 1 1 1 0 0 1 1 0 1 0 1 1 1 1 0 1 0 0 1 0 1 1 0 1 0]"
#bit = "01011010010111101011001110010001"
#print(bit.count('1'))
#print(bit.count('0'))

def neuron_counter(in_dim = 3, layer_counts: tuple = (256,)*5, head_counts = (), d_out = 1, c1_out = 32, c2_out = 32):
    depth = len(layer_counts)
    total = (in_dim+1) * layer_counts[0] 
    for i in range(1, depth-1):
        total += (layer_counts[i-1]+1) * layer_counts[i] 

    head_counts = (layer_counts[-1],) + head_counts
    for i in range(1, len(head_counts)):
        total += 3*(head_counts[i-1]+1) * head_counts[i]
    total += head_counts[-1] * (d_out + c1_out + c2_out)
    
    return f"n: {total} | {round(total*0.00004, 2)} mb" 

width = 64
print(neuron_counter(
    layer_counts=(width,)*54
    #, head_counts=(width,)
    ))