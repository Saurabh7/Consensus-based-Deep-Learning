# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 00:49:02 2020

@author: Nitin

simulation.cycles 500

"""

## File to create config files automatically for the respective datasets

import pandas as pd
import os
import numpy as np
from copy import copy
from collections import Counter
import logging
import random
import math



# constants
DATA_DIR = "../data/"

def load_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_gen_file", type=str,
                        help="path of config gen file")
    args = parser.parse_args()
    return args


def load_base_config(path):
    with open(path, "r") as f:
        base_config_lines = f.readlines()
        return base_config_lines
    return []


def append_to_base_config(base_config_lines, config):
    for key in config.keys():
        base_config_lines.append(key +" " + str(config[key]))
    return base_config_lines
    
    
def load_config_gen_file(path):
    from collections import defaultdict
    config_dict = defaultdict(dict)
    with open(path, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            config_values = line.lstrip().rstrip().split(",")
            print(config_values)
            assert len(config_values) == 13
            
            
            config_dict["config_{}".format(i)]["simulation.cycles"] = config_values[0]
            config_dict["config_{}".format(i)]["network.size"] = config_values[1]
            config_dict["config_{}".format(i)]["network.node.resourcepath"] = config_values[2]
            config_dict["config_{}".format(i)]["network.node.trainlen"] = config_values[3]
            config_dict["config_{}".format(i)]["network.node.testlen"] = config_values[4]
            
            config_dict["config_{}".format(i)]["network.node.numhidden"] = config_values[5]
            config_dict["config_{}".format(i)]["network.node.learningrate"] = config_values[6]
            config_dict["config_{}".format(i)]["network.node.cyclesforconvergence"] = config_values[7]
            config_dict["config_{}".format(i)]["network.node.convergenceepsilon"] = config_values[8]
            config_dict["config_{}".format(i)]["network.node.lossfunction"] = config_values[9]
            config_dict["config_{}".format(i)]["network.node.hidden_layer_act"] = config_values[10]
            config_dict["config_{}".format(i)]["network.node.final_layer_act"] = config_values[11]
            
            # needed just for the sake of the code
            config_dict["config_{}".format(i)]["dataset"] = config_values[12]
            
    return config_dict


def generate_configs_from_dict():
    
    params_dict = {"learning_rate": [0.001],
                   "batch_size": [1],
                   "init_method":[0],
                   "numhidden_1":[50],
                   "numhidden_2": [20],
                   "numhidden_3": [10],
                   "cycles_for_convergence":[50000],
                   "convergence_epsilon":[0.0001],
                   "random.seed":[10],
                   "loss_function": ["cross_entropy"],
                   "hidden_layer_act": ["relu"],
                   "final_layer_act": ["softmax"],
                   "feature_split_type": ["random"],
                   "overlap_ratio":[0],
                   "nn_type":["mlp"],
                   "num_layers":[2],
                   "dataset_name":["mnist_balanced"],
                   "network.size": [10],
                   "simulation.cycles": [4000],
                   "degree":[2]
                   }
    from itertools import product
    keys, values = zip(*params_dict.items())
    all_perms = [dict(zip(keys, v)) for v in product(*values)]
    print(len(all_perms))
    
    # subset by conditions
    updated_perms = []
    run_dict = {}
    
    for i, perm in enumerate(all_perms):

        if perm["num_layers"] in [1,2]:
#            if perm["numhidden_1"] > perm["numhidden_2"]:
            if perm["dataset_name"] not in run_dict.keys():
                run_dict[perm["dataset_name"]] = 1
            else:
                run_dict[perm["dataset_name"]] += 1

            perm["run"] = run_dict[perm["dataset_name"]]
            perm["resourcepath"] = perm["dataset_name"]
            if perm["feature_split_type"] == "overlap":
                perm["overlap_ratio"] = 0.2
            updated_perms.append(perm)
    
    
#    params_dict_1h = {"learning_rate": [0.01, 0.0001],
#                   "batch_size": [1],
#                   "init_method":[0],
#                   "numhidden_1":[25, 50, 100],
#                   "numhidden_2": [10],
#                   "numhidden_3": [10],
#                   "cycles_for_convergence":[20],
#                   "convergence_epsilon":[0.0001],
#                   "random.seed":[10, 234, 4657],
#                   "loss_function": ["cross_entropy"],
#                   "hidden_layer_act": ["relu", "tanh", "sigmoid"],
#                   "final_layer_act": ["softmax"],
#                   "feature_split_type": ["random"],
#                   "overlap_ratio":[0],
#                   "nn_type":["mlp"],
#                   "num_layers":[1],
#                   "dataset_name":["dexter", "gisette"],
#                   "network.size": [1, 10],
#                   "simulation.cycles": [3000]
#                   }
#    
#    keys, values = zip(*params_dict_1h.items())
#    all_perms_1h = [dict(zip(keys, v)) for v in product(*values)]
#    print(len(all_perms_1h))
#
#    
#    for i, perm in enumerate(all_perms):
#        if perm["num_layers"] == 1:
#            if perm["numhidden_1"] > perm["numhidden_2"]:
#                if perm["dataset_name"] not in run_dict.keys():
#                    run_dict[perm["dataset_name"]] = 1
#                else:
#                    run_dict[perm["dataset_name"]] += 1
#    
#                perm["run"] = run_dict[perm["dataset_name"]]
#                perm["resourcepath"] = perm["dataset_name"]
#                updated_perms.append(perm)
    
#    # Remove all second layer perms if num_layers 1
#    perms_with_1h = [perm for perm in all_perms if perm["num_layers"] == 1]
#    for i in range(len(perms_with_1h)):
#        perms_with_1h[i]["numhidden_2"] = 10 # Set the same value for num_hidden2
#    
#    print(len(updated_perms))
#    print(len(perms_with_1h))
#    perms_with_1h = [dict(y) for y in set(tuple(x.items()) for x in perms_with_1h)]
#    print(len(perms_with_1h))
#    
#    new_perms_with_1h = []
#    for i,perm in enumerate(perms_with_1h):
#        if perm["dataset_name"] not in run_dict.keys():
#            run_dict[perm["dataset_name"]] = 1
#        else:
#            run_dict[perm["dataset_name"]] += 1
#
#        perm["run"] = run_dict[perm["dataset_name"]]
#        perm["resourcepath"] = perm["dataset_name"]
#        new_perms_with_1h.append(perm)
        
    
#    updated_perms += new_perms_with_1h
    
    print(len(updated_perms))
    return updated_perms, run_dict
    

if __name__ == "__main__":
    perms, run_dict = generate_configs_from_dict()
    
#    args = load_args()
#    config_dict = load_config_gen_file(args.config_gen_file)
#    
#    # Create a file for each of these
    
    
    for dataset in run_dict.keys():
        op_dir = os.path.join("../config", dataset)
        if os.path.exists(op_dir):
            import shutil
            shutil.rmtree(op_dir)
        os.makedirs(op_dir)
        
    for perm in perms:
        base_config_lines = load_base_config("../config/base_config.cfg")
        updated_config = append_to_base_config(base_config_lines, perm)
        
        
        op_dir = os.path.join("../config", perm["dataset_name"])
        with open(os.path.join(op_dir, "config_{}.cfg".format(perm["run"])), "w") as f:  
            f.write("\n".join(updated_config))
#        
#
