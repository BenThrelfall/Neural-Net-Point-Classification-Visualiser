# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 15:31:49 2020

@author: Owner
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import time
import json

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from datetime import datetime


def get_output(model):
    
    grid_min = -0.5
    grid_max = 1.5

    grid_width = 100

    number_line = np.arange(grid_min * grid_width,grid_max * grid_width,1)

    grid = []
    for x in number_line:
        for y in reversed(number_line):
            grid.append((x/grid_width,y/grid_width))
    
    grid_array = np.array(grid)
    test_data = torch.from_numpy(grid_array).float()
    
    output = model(test_data)
    output = torch.sigmoid(output)
    
    return output


def make_output_grid(output):
    
    grid_min = -0.5
    grid_max = 1.5

    grid_width = 100

    number_line = np.arange(grid_min * grid_width,grid_max * grid_width,1)
    
    output_grid = np.zeros((len(number_line),len(number_line)))
    i = 0
    for x in range(len(number_line)):
        for y in range(len(number_line)):
            output_grid[y,x] = (output[i])
            i = i + 1
            
    return output_grid


def make_output_grid_from_tensor(output):
    
    grid_min = -0.5
    grid_max = 1.5

    grid_width = 100

    number_line = np.arange(grid_min * grid_width,grid_max * grid_width,1)
    
    output_grid = np.zeros((len(number_line),len(number_line)))
    i = 0
    for x in range(len(number_line)):
        for y in range(len(number_line)):
            output_grid[y,x] = (output[i].item())
            i = i + 1
            
    return output_grid


def points_to_serializable(points):
    output = []
    for point in points:
        output.append((point[0],point[1]))
    
    return output


def output_to_serializable(output):
    array = []
    for number in output:
        array.append(number.item())
        
    return array


def labels_to_serializable(labels):
    array = []
    for number in labels:
        array.append(int(number[0]))
        
    return array

def train(model, criterion, optimizer, points, labels, epoches=5, do_print=True):
    
    total_start_time = time.time()
    loss = -1
    for i in range(1,epoches+1):
        start_time = time.time()
        optimizer.zero_grad()
        y = model(points)
        loss = criterion(y, labels)
        loss.backward()
        optimizer.step()
        end_time = time.time()
        if(i % 1000 == 0 and do_print):
            print(f"Loss {loss.item()}, Time {end_time - start_time}")
            
    total_end_time = time.time()
    total_time_taken = total_end_time - total_start_time
    if(do_print):
        print(f"Total Time {total_time_taken}")
        
    final_loss = loss.item()
    
    return (total_time_taken, final_loss)