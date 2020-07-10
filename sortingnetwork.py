# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 15:15:56 2020

@author: Owner
"""

from torch import nn

class SortingNetwork(nn.Module):
    
    def __init__(self, nodes, num_layers=1, activation_function=nn.Sigmoid):
        super().__init__()
        
        nodes_per_layer = nodes // num_layers
        assert(nodes_per_layer >= 1)
        remainder = nodes % num_layers
        
        layers = []
        previous_layer = 2
        for i in range(num_layers):
            nodes_for_layer = nodes_per_layer
            if remainder > 0:
                nodes_for_layer += 1
                remainder -= 1
            layers.append(nn.Linear(previous_layer, nodes_for_layer))
            layers.append(activation_function())
            previous_layer = nodes_for_layer
            
        self.layers = nn.Sequential(*layers)
        self.output = nn.Linear(previous_layer, 1)
        
    def forward(self, x):
        
        x = self.layers(x)
        x = self.output(x)
        
        return x
        