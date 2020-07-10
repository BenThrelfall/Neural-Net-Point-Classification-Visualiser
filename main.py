# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 14:56:09 2020

@author: Owner
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import json

import torch
from torch import nn, optim

from datetime import datetime

from helpers import get_output, make_output_grid, points_to_serializable, output_to_serializable, labels_to_serializable, train
from sortingnetwork import SortingNetwork

def main():
    
    def menu_selection(menu_items):
        user_input = ""
        check_values = set([str(i) for i in range(1,len(menu_items) + 1)])
        while user_input not in check_values:
            user_input = input("> ")
        
        return menu_items[int(user_input)-1]

    def generate_points_randomly():
        print("How many points should be generated?")
        user_input = ""
        while not user_input.isnumeric():
            user_input = input("> ")
        points, labels = create_random_points_and_labels(int(user_input))
        return (points, labels)
    
    def input_points_manually():
        print("points should be between 0 and 1")
        print("Labels should be 0 or 1")
        points = []
        labels = []
        i = 0
        while True:
            x = choose_float(f"x{i}") 
            y = choose_float(f"y{i}")
            label = choose_integer(f"label{i}")
            point = (x,y)
            points.append(point)
            labels.append(label)
            i += 1
            
            print("Done?")
            done = input("> ")
            if done in ['Done','done']:
                break
            
        
        points = np.array(points)
        labels = np.array(labels)
        
        return (points, labels)
    
    #This might not work - test at somepoint
    def load_points_from_file():
        print("enter points file path")
        file_path = input("> ")
    
        points = []
        
        with open(file_path, "r") as fb:
            points = json.loads(fb.read())
            
        print("enter labels file path")
        file_path = input("> ")
    
        labels = []
        
        with open(file_path, "r") as fb:
            labels = json.loads(fb.read())
            
        points = np.array(points)
        labels = np.array(labels)
        labels = labels.reshape(len(labels),1)
        
        print(f"loaded points \n {points}")
        print(f"loaded labels \n {labels}")
        
        return (points, labels)
    
    def choose_learning_rate():
        print("What learning rate should be used?")
        user_input = ""
        while not isFloat(user_input):
            user_input = input("> ")
        return float(user_input)
    
    def choose_float(message):
        print(message)
        user_input = ""
        while not isFloat(user_input):
            user_input = input("> ")
        return float(user_input)
    
    def choose_integer(message):
        print(message)
        user_input = ""
        while not user_input.isnumeric():
            user_input = input("> ")
        return int(user_input)
    
    def isFloat(s):
        try:
            float(s)
            return True
        except ValueError:
            return False
    
    def passFunction(*args, **kwargs):
        pass
    
    def raiseNotImplemented(*args, **kwargs):
        raise NotImplementedError()
    
    def firstMenu():
        print("===Mass, Point Sorting Network, Tester===")
        print("Enter 1 or 2")
        print("1 - start new testing run")
        print("2 - load data from existing JSON")
        
        menu_selection([doNewRun, loadFromExisting])()
        
    def doNewRun():     
        points, labels = pointsCreationMenu()
        displayPointsMenu(points, labels)
        load_from_path, save_prefix = massNetworkTestingMenus(points, labels)
        visualizationMenu(load_from_path, save_prefix)
    
    def loadFromExisting():
        print("Enter path to JSON file")
        load_from_path = input("> ")
        print("Enter path to save figure at")
        save_prefix = input("> ")
        visualizationMenu(load_from_path, save_prefix)
    
    def pointsCreationMenu():
        print("1 - generate random points and labels")
        print("2 - load points and labels from file")
        print("3 - enter points and labels manually")
        
    
        points_function = menu_selection([generate_points_randomly,
                                          load_points_from_file,
                                          input_points_manually])
        
        points, labels = points_function()
        return (points, labels)
    
    def displayPointsMenu(points, labels):
        print("1 - display points")
        print("2 - continue")
        
        menu_selection([display_points, passFunction])(points, labels)
    
    def massNetworkTestingMenus(points, labels):
        print("Enter save file prefix (file paths will work)")
        save_prefix = input("> ")
        
        print("Choose hyperparameters for neural networks")
        print("Choose optimizer")
        print("1 - SGD")
        print("2 - Adam")
        print("3 - Adagrad")
        print("4 - Adadelta")
        
        optimizer = menu_selection([optim.SGD,
                                    optim.Adam,
                                    optim.Adagrad,
                                    optim.Adadelta])
        
        learning_rate = choose_learning_rate()
        
        criterion = nn.BCEWithLogitsLoss
        
        print("Choose activation function")
        print("1 - ReLU")
        print("2 - Sigmoid")
        print("3 - Tanh")
        print("4 - Hardtanh")
        
        activation_function = menu_selection([nn.ReLU,
                                              nn.Sigmoid,
                                              nn.Tanh,
                                              nn.Hardtanh])
        
        epochs = choose_integer("How many epochs should the networks be trained for?")
        
        seed = choose_integer("What random seed should be used?")
        
        while True:
            node_domain_min = choose_integer("How many hidden nodes should the smallest network have?")
            node_domain_max = choose_integer("How many hidden nodes should the largest network have?")
            if(node_domain_min < node_domain_max):
                break
        
        while True:
            layer_domain_min = choose_integer("How many hidden layers should the smallest network have?")
            layer_domain_max = choose_integer("How many hidden layers should the largest network have?")
            if(node_domain_min < node_domain_max):
                break
        
        load_from_path = createNNDataArray(points, labels, epochs, learning_rate, criterion,
                          optimizer, activation_function, node_domain_min,
                          node_domain_max, layer_domain_min, layer_domain_max,
                          seed=seed, filename=save_prefix + "_data.json")
        
        return (load_from_path, save_prefix)
    
    def visualizationMenu(load_from_path, save_prefix):
        print("==================")
        print("1 - create visualization")
        print("2 - exit")
        
        menu_selection([create_and_save_visualisation, passFunction])(loadfilepath=load_from_path,
                                                                      savefilepath=save_prefix + "_figure.png")
        
    firstMenu()
        
def create_random_points_and_labels(amount=26):
   
    points = np.random.rand(amount,2)
    labels = np.zeros((amount,1))
    
    labels[0 : len(points)//2] = 0
    labels[len(points)//2 : len(points)] = 1
    
    return (points, labels)
    
def display_points(points, labels):
    red = []
    blue = []
    
    for i, point in enumerate(points):
        if labels[i] == 1:
            red.append(point)
        else:
            blue.append(point)
            
    blue = np.array(blue)
    red = np.array(red)

    plt.scatter(red[:,0], red[:,1], c="red")
    plt.scatter(blue[:,0], blue[:,1], c="blue")
    plt.show()

def createNNDataArray(points, labels, epochs, learning_rate, criterion,
                      optimiser, activation_function, node_domain_min,
                      node_domain_max, layer_domain_min, layer_domain_max,seed=42, filename="all_outputs.json"):
    all_network_output_datas = []
    
    points_tensor = torch.from_numpy(points).float()
    labels_tensor = torch.from_numpy(labels).float()
    
    for n_layers in range(layer_domain_min, layer_domain_max+1):
        for n_nodes in range(node_domain_min, node_domain_max+1):
            if(n_nodes >= n_layers):
                print(f"Starting {n_nodes} node, {n_layers} layer network")
                torch.manual_seed(seed)
                model = SortingNetwork(n_nodes, n_layers, activation_function)
                optimizer = optimiser(model.parameters(),lr=learning_rate)
                crit = criterion()
                
                time_taken, final_loss = train(model, crit, optimizer, points_tensor, labels_tensor, epochs, do_print=False)
                output = get_output(model)
                output = output_to_serializable(output)
                
                current_data = {'points': points_to_serializable(points), 'labels': labels_to_serializable(labels),
                                'output': output, 'training_time': time_taken,
                                'loss': final_loss,'epochs': epochs, 'layers': n_layers,
                                'nodes': n_nodes, 'network': str(model),
                                'optimiser':str(optimizer), 'criterion':str(crit)}
                all_network_output_datas.append(current_data)
    
    return_file_path = ""
    
    try:
        with open(filename, 'w') as fp:       
            json.dump(all_network_output_datas, fp)
            return_file_path = filename
    except:
        temp_file_name = "output_data" + str(datetime.now()).replace(':','') + ".json"
        print(f"File failed to write at {filename}. Writting to {temp_file_name}")
        try:
            with open(temp_file_name, 'w') as fp:       
                json.dump(all_network_output_datas, fp)
                return_file_path = temp_file_name
        except:
            print("Failed to write file again. ABORTING")
            return
        
        
    print("Data saved")
    return return_file_path

def create_and_save_visualisation(cols = 5, loadfilepath='all_outputs.json', savefilepath=None):
    
    with open(loadfilepath, 'r') as fp:       
        all_data = json.loads(fp.read())
    
    rows = (len(all_data) // cols) + 1
    
    fig, axs = plt.subplots(ncols=cols, nrows=rows, figsize=(cols*12,rows*12))
    for i, network_data in enumerate(all_data):
        cur_axs = axs[i//cols,i%cols]
        sb.heatmap(make_output_grid(network_data['output']), cbar=False, xticklabels=False, yticklabels=False,
                   cmap="seismic", square=True, vmin=0, vmax=1, ax=cur_axs)
        
        cur_axs.set_title("n {0}; l {1}; loss {2:.3f}; t: {3:.3f}"
                                .format(network_data['nodes'],network_data['layers'],
                                        network_data['loss'],network_data['training_time']),
                         fontdict={'fontsize': 40})
    if savefilepath != None:
        fig.savefig(savefilepath)
        try:
            fig.savefig(savefilepath)
        except:
            temp_file_name = "figure" + str(datetime.now()).replace(':','') + ".png"
            print(f"File failed to write at {savefilepath}. Writting to {temp_file_name}")
            try:
                fig.savefig(temp_file_name)
            except:
                print("Failed to write file again. ABORTING")
                return
    else:
        temp_file_name = "figure" + str(datetime.now()).replace(':','') + ".png"
        fig.savefig(f"figure.png")

    

if __name__ == '__main__':
    main()