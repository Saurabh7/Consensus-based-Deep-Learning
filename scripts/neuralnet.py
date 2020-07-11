# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 20:46:56 2020

@author: nitin
"""

# Create non-linearly separable data

import numpy as np
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, log_loss
import random
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import make_blobs

base_dir = "C:/Users/nitin/eclipse-workspace/consensus-deep-learning-version-2.0/data"
import torch
import warnings
import os
import logging
import gc

import torch.nn.functional as F
warnings.filterwarnings('ignore')

# Todo: data loading should be done in another class
# Todo: Make each model into a class and pass this into higher NNTrainer class



def get_tensors_in_memory():
    tensor_count = 0
    total_size = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                total_size += obj.size()
                tensor_count += 1
        except:
            pass
    return tensor_count, total_size
        

def softmax(t):
    return t.exp()/t.exp().sum(-1).unsqueeze(-1)
    
def roc_auc_compute_fn(y_preds, y_targets):
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        raise RuntimeError("This contrib module requires sklearn to be installed.")

    y_true = y_targets.detach().numpy()
    y_pred = y_preds.detach().numpy()
    return roc_auc_score(y_true, y_pred)


class NeuralNetworkCluster:
    """
    Holds several neural networks in a dictionary.
    Each NN corresponds to a node in the distributed algorithm.
    """
    def __init__(self, base_dir):
        from collections import defaultdict
        self.neuralNetDict = defaultdict(dict)
        self.base_dir = base_dir
        # This will store feature indices for each node - determined by overlap functionality
        self.featureDict = {1: []}
        self.epoch = 0

    def init_data(self, nn_config):
        self.nn_config = nn_config
        random.seed(int(nn_config["random.seed"]))
        train_filename = "{}_{}.csv".format(nn_config["dataset_name"], "train_binary")
        test_filename = "{}_{}.csv".format(nn_config["dataset_name"], "test_binary")
        #
        self.convergence_epsilon = nn_config["convergence_epsilon"]
        self.convergence_iters = nn_config["cycles_for_convergence"]
        #
        self.df_train = pd.read_csv(os.path.join(self.base_dir, nn_config["dataset_name"], train_filename))
        self.labels_train = self.df_train.pop('label')
        self.df_test = pd.read_csv(os.path.join(self.base_dir, nn_config["dataset_name"], test_filename))
        self.labels_test = self.df_test.pop('label')

        idx_dict = {}
        num_nodes = int(nn_config["num_nodes"])
        if nn_config["feature_split_type"] == "random":
            used_indices = []
            num_features = len([col for col in self.df_train.columns if col not in ['label']])
            num_features_split = int(np.ceil(num_features / float(num_nodes)))

            for split in range(num_nodes):
                # if split == numsplits - 1:
                #   num_features_split = num_features - len(used_indices)

                remaining_indices = [i for i in range(num_features) if i not in used_indices]
                print(len(used_indices), len(remaining_indices))
                try:
                    idx = random.sample(remaining_indices, num_features_split)
                except ValueError:
                    # if the last split has lesser number of indices than what is being sampled for
                    idx = remaining_indices
                idx_dict[split] = idx
                used_indices = used_indices + idx

        elif nn_config["feature_split_type"] == "overlap":
            used_indices = []
            num_features = len([col for col in self.df_train.columns if col not in ['label']])
            num_features_split = int(np.ceil(num_features / float(num_nodes)))
            num_features_overlap = int(np.ceil(nn_config["overlap_ratio"] * num_features_split))
            num_features_split = num_features_split - num_features_overlap

            overlap_features = random.sample([i for i in range(num_features)], num_features_overlap)
            used_indices += overlap_features
            for split in range(num_nodes):
                # if split == numsplits - 1:
                #   num_features_split = num_features - len(used_indices)

                remaining_indices = [i for i in range(num_features) if i not in used_indices]
                idx = random.sample(remaining_indices, num_features_split)
                idx_dict[split] = idx + overlap_features
                used_indices = used_indices + idx
        
        elif nn_config["feature_split_type"] == "spatial":
            print("Inside spatial")
            # Spatial splitting of mnist-balanced dataset - just need to determine feature indices
            assert nn_config["dataset_name"] == "mnist_balanced", "spatial splits only possible with mnist dataset"
            assert num_nodes == 16, "spatial splits only possible with 16 nodes currently"
            # create an index matrix to know which indices are being subset
            split_num = 0
            index_mat = np.arange(784)
            index_mat = index_mat.reshape((28, 28))
            for i in range(4):
                for j in range(4):
                    indices = index_mat[i*7:(i+1)*7, j*7:(j+1)*7]
                    idx_dict[split_num] = indices.reshape((-1, 49)).squeeze()
                    split_num += 1
            
        self.feature_dict = idx_dict

    def appendNNToCluster(self, nn_config):
        node_id = nn_config["node_id"]
        print("Inside appendNNToCluster")
#        start_tensors, start_size = get_tensors_in_memory()   
        print("Appending node_id {} to cluster".format(node_id))

        if node_id in self.neuralNetDict.keys():
            logging.info("node_id: {} already exists in dictionary. Overwriting...".format(node_id))

        if nn_config["num_layers"] == 1:
            model = SingleLayerNeuralNetwork()
            df_train_node = self.df_train.iloc[:, self.feature_dict[node_id]]
            df_test_node = self.df_test.iloc[:, self.feature_dict[node_id]]
            model.set_data(df_train_node, self.labels_train, df_test_node, self.labels_test)
            model.initialize(nn_config)

        if nn_config["num_layers"] == 2:
            model = TwoLayerNeuralNetwork()
            df_train_node = self.df_train.iloc[:, self.feature_dict[node_id]]
            df_test_node = self.df_test.iloc[:, self.feature_dict[node_id]]
            model.set_data(df_train_node, self.labels_train, df_test_node, self.labels_test)
            model.initialize(nn_config)

        if nn_config["num_layers"] == 3:
            model = ThreeLayerNeuralNetwork()
            df_train_node = self.df_train.iloc[:, self.feature_dict[node_id]]
            df_test_node = self.df_test.iloc[:, self.feature_dict[node_id]]
            model.set_data(df_train_node, self.labels_train, df_test_node, self.labels_test)
            model.initialize(nn_config)

        self.neuralNetDict[node_id]["model"] = model

        # Loss criterion
        if nn_config["loss_function"] == "cross_entropy":
            criterion = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError("{} is not a supported loss function".format(nn_config["loss_function"]))
        self.neuralNetDict[node_id]["criterion"] = criterion

        # Optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=nn_config["learning_rate"])
        self.neuralNetDict[node_id]["optimizer"] = optimizer

        self.neuralNetDict[node_id]["train_losses"] = []
        self.neuralNetDict[node_id]["train_accuracy"] = []
        self.neuralNetDict[node_id]["test_losses"] = []
        self.neuralNetDict[node_id]["test_accuracy"] = []
        self.neuralNetDict[node_id]["train_auc"] = []
        self.neuralNetDict[node_id]["test_auc"] = []
        self.neuralNetDict[node_id]["converged_iters"] = 0
        self.neuralNetDict[node_id]["converged_states"] = []
        self.neuralNetDict[node_id]["converged_flag"] = False
        self.neuralNetDict[node_id]["overall_train_accuracy"] = []
        self.neuralNetDict[node_id]["overall_test_accuracy"] = []
        self.neuralNetDict[node_id]["overall_train_auc"] = []
        self.neuralNetDict[node_id]["overall_test_auc"] = []
        self.neuralNetDict[node_id]["converged_flag"] = "false"
        self.neuralNetDict[node_id]["converged_flags"] = []
        self.neuralNetDict[node_id]["prev_loss"] = None
        self.save_results = True
        
#        end_tensors, end_size = get_tensors_in_memory()
#        print("Number of tensors added in appendNNToCluster: {}, size added: {}".format(end_tensors - start_tensors, end_size-start_size))
        
    def gossip(self, node_id, neighbor_node_id):
        """
        Performs gossip on two given node_ids.
        """
        print("Inside gossip")
#        start_tensors, start_size = get_tensors_in_memory()
        if self.neuralNetDict[node_id]["converged_iters"] >= self.convergence_iters:
            print("Node {} has converged.".format(node_id))
            self.neuralNetDict[node_id]["converged_flag"] = "true"
            return
        print("node_id", node_id)
        model0 = self.neuralNetDict[node_id]["model"]
        model1 = self.neuralNetDict[neighbor_node_id]["model"]
        
        criterion0 = self.neuralNetDict[node_id]["criterion"]
        criterion1 = self.neuralNetDict[neighbor_node_id]["criterion"]
        
        optimizer0 = self.neuralNetDict[node_id]["optimizer"]
        optimizer1 = self.neuralNetDict[neighbor_node_id]["optimizer"]
        
        # Wipe gradients of both optimizers
        optimizer0.zero_grad()
        optimizer1.zero_grad()
        
        # Forward pass
        y_pred0 = model0(model0.X_train)
        y_pred1 = model1(model1.X_train)
        
        y_pred0_2 = y_pred0.clone()
        y_pred1_2 = y_pred1.clone()
        
        y_pred_mean0 = (y_pred0 + y_pred1)/2
        y_pred_mean1 = (y_pred0_2 + y_pred1_2)/2
        
        # Compute Loss
        loss0 = criterion0(y_pred_mean0.squeeze(), model0.y_train)
        loss1 = criterion1(y_pred_mean1.squeeze(), model1.y_train)
        
        ## If the abs diff between current loss and previous loss < convergence_epsilon
        if self.neuralNetDict[node_id]["prev_loss"] is None:
            self.neuralNetDict[node_id]["converged_iters"] = 0
        else:
            diff = abs(loss0.item() - self.neuralNetDict[node_id]["prev_loss"])
            if diff < self.convergence_epsilon:
                self.neuralNetDict[node_id]["converged_iters"] += 1
                
            else:
                self.neuralNetDict[node_id]["converged_iters"] = 0
        self.neuralNetDict[node_id]["prev_loss"] = loss0.item()
        # Backward pass
        loss0.backward(retain_graph=True)
        loss1.backward(retain_graph=True)
        
        optimizer0.step()
        optimizer1.step()
#        
#        # Clear all local variables
#        del y_pred0, y_pred1, y_pred0_2, y_pred1_2
#        del y_pred_mean0, y_pred_mean1
#        del loss0, loss1
#        del optimizer0, optimizer1
#        gc.collect()
#        
#        end_tensors, end_size = get_tensors_in_memory()
#        print("Number of tensors added in gossip: {}, size added: {}".format(end_tensors - start_tensors, end_size-start_size))
#        print("Train Loss @ Node {}: {}, Train Loss @ Node {}: {}".format(node_id, 
#              loss0.item(), neighbor_node_id, loss1.item()))
        
    def save_results(self):
        """
        Stores a pickle file of the NeuralNetworkCluster object
        """
        
        import pickle
        pickle.dump()
    
    def compute_losses_and_accuracies(self):
        """
        Computes train and test losses for all the nodes.
        """
        print("Inside compute_losses_and_accuracies")
#        start_tensors, start_size = get_tensors_in_memory()
        
        y_pred_train_agg = []
        y_pred_test_agg= []
        print("Calculating losses: ")
        
        for node_id in self.neuralNetDict.keys():
            model = self.neuralNetDict[node_id]["model"]
            criterion = self.neuralNetDict[node_id]["criterion"]        
            
            # Compute Train Loss
            y_pred_train = model(model.X_train)
#            print(type(y_pred_train))
            y_pred_train = y_pred_train.squeeze()
            train_loss = criterion(y_pred_train, model.y_train) 
            train_output = (y_pred_train[:, 1]>0.5).float()
            train_correct = (train_output == model.y_train).float().sum()
            train_accuracy = train_correct/model.X_train.shape[0]
            
            train_auc_score = roc_auc_compute_fn(y_pred_train[:, 1], model.y_train)
            self.neuralNetDict[node_id]["train_losses"].append(train_loss.item())
            self.neuralNetDict[node_id]["train_accuracy"].append(train_accuracy.item())
            self.neuralNetDict[node_id]["train_auc"].append(train_auc_score.item())
            
            # Compute Test Loss
            y_pred_test = model(model.X_test)
            y_pred_test = y_pred_test.squeeze()
            test_loss = criterion(y_pred_test, model.y_test)
            test_output = (y_pred_test[:, 1]>0.5).float()
            test_correct = (test_output == model.y_test).float().sum()
            test_accuracy = test_correct/model.X_test.shape[0]

            test_auc_score = roc_auc_compute_fn(y_pred_test[:, 1], model.y_test) 
            self.neuralNetDict[node_id]["test_losses"].append(test_loss.item())
            self.neuralNetDict[node_id]["test_accuracy"].append(test_accuracy.item())
            self.neuralNetDict[node_id]["test_auc"].append(test_auc_score.item())
            
            self.neuralNetDict[node_id]["converged_states"].append(self.neuralNetDict[node_id]["converged_iters"])
            self.neuralNetDict[node_id]["converged_flags"].append(self.neuralNetDict[node_id]["converged_flag"])
            

            y_pred_train_agg.append(y_pred_train[:,1].float())
            y_pred_test_agg.append(y_pred_test[:,1].float())
            
#            del y_pred_train, train_loss, train_output, train_correct, train_accuracy, train_auc_score
#            del y_pred_test, test_loss, test_output, test_correct, test_accuracy, test_auc_score
        
        # Obtain average predictions
        y_pred_train_agg_pyt = torch.stack(y_pred_train_agg, 0)
        y_pred_train_agg_pyt = torch.mean(y_pred_train_agg_pyt, 0)
        overall_train_output = (y_pred_train_agg_pyt>0.5).float()
        overall_train_correct = (overall_train_output == model.y_train).float().sum()
        overall_train_accuracy = overall_train_correct/model.X_train.shape[0]
        overall_train_auc = roc_auc_compute_fn(y_pred_train_agg_pyt, model.y_train)

        y_pred_test_agg_pyt = torch.stack(y_pred_test_agg, 0)
        y_pred_test_agg_pyt = torch.mean(y_pred_test_agg_pyt, 0)
        overall_test_output = (y_pred_test_agg_pyt>0.5).float()
        overall_test_correct = (overall_test_output == model.y_test).float().sum()
        overall_test_accuracy = overall_test_correct/model.X_test.shape[0]
        overall_test_auc = roc_auc_compute_fn(y_pred_test_agg_pyt, model.y_test)
        
        
        print("Overall Train AUC: {}, Overall Test AUC: {}".format(overall_train_auc.item(), overall_test_auc.item()))
        for node_id in self.neuralNetDict.keys():
            self.neuralNetDict[node_id]["overall_train_accuracy"].append(overall_train_accuracy.item())
            self.neuralNetDict[node_id]["overall_test_accuracy"].append(overall_test_accuracy.item())
            self.neuralNetDict[node_id]["overall_train_auc"].append(overall_train_auc.item())
            self.neuralNetDict[node_id]["overall_test_auc"].append(overall_test_auc.item())
            
        del overall_train_output, overall_train_correct, overall_train_accuracy, overall_train_auc
        del overall_test_output, overall_test_correct, overall_test_accuracy, overall_test_auc
        del y_pred_train_agg_pyt, y_pred_test_agg_pyt
        
#        torch.cuda.empty_cache()
#        
#        end_tensors, end_size = get_tensors_in_memory()
#        print("Number of tensors added in compute_losses_and_accuracies: {}, size added: {}".format(end_tensors - start_tensors, end_size-start_size))

    def train(self, node_id):
        """
        Used for training on only one node in centralized execution. 
        No gossip is performed here.
        """
        
        if self.neuralNetDict[node_id]["converged_iters"] >= self.convergence_iters:
            print("Node {} has converged.".format(node_id))
            self.neuralNetDict[node_id]["converged_flag"] = "true"
            return
        print("node_id", node_id)
        model0 = self.neuralNetDict[node_id]["model"]
        criterion0 = self.neuralNetDict[node_id]["criterion"]        
        optimizer0 = self.neuralNetDict[node_id]["optimizer"]        
        # Wipe gradients of both optimizer
        optimizer0.zero_grad()
        # Forward pass
        y_pred0 = model0(model0.X_train)        
        # Compute Loss
        loss0 = criterion0(y_pred0.squeeze(), model0.y_train)
        
        
        ## If the abs diff between current loss and previous loss < convergence_epsilon
        if self.neuralNetDict[node_id]["prev_loss"] is None:
            self.neuralNetDict[node_id]["converged_iters"] = 0
        else:
            diff = abs(loss0.item() - self.neuralNetDict[node_id]["prev_loss"])
            if diff < self.convergence_epsilon:
                self.neuralNetDict[node_id]["converged_iters"] += 1
                
            else:
                self.neuralNetDict[node_id]["converged_iters"] = 0
        self.neuralNetDict[node_id]["prev_loss"] = loss0.item()

        # Backward pass
        loss0.backward(retain_graph=True)
        # Update parameters
        optimizer0.step()
        
        

class SingleLayerNeuralNetwork(torch.nn.Module):
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.nn_config_dict = {}
    
    def initialize(self, nn_config_dict):
        self.nn_config_dict = nn_config_dict
        super(SingleLayerNeuralNetwork, self).__init__()
        self.input_size = self.X_train.shape[1]
        self.hidden_size  = nn_config_dict["numhidden_1"]
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        
        # Define the activation functions to be used
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = softmax
        # self.batchnorm1 = torch.nn.BatchNorm1d(self.hidden_size)
        self.dropout = torch.nn.Dropout(p=0.1)
        
        self.fc2 = torch.nn.Linear(self.hidden_size, 2)
        # Define hidden layer and final layer activastion functions
        self.hidden_act_func = self.get_hidden_act_function()
        self.final_act_func = self.get_final_act_function()
    
#     def load_data(self, dataset, base_dir, feature_split, node_id=None):
#         df_train = pd.read_csv(os.path.join(base_dir,
#                                       dataset,
#                                       "feature_split_" + str(feature_split),
#                                      train_filename), header=None).to_numpy()
#
#
#         df_test = pd.read_csv(os.path.join(base_dir,
#                                       dataset,
#                                       "feature_split_" + str(feature_split),
#                                       test_filename), header=None).to_numpy()
#
#
#         # Split into features and labels
#         X_train = df_train[:,1:]
#         y_train = df_train[:,0]
#
#         X_test= df_test[:,1:]
#         y_test = df_test[:,0]
#
#
#         # Map to tensors so torch can use this data
#         X_train, y_train, X_test, y_test = map(torch.tensor, (X_train, y_train, X_test, y_test))
#
#         # Convert features to float and labels to long
#         self.X_train = X_train.float()
#         self.y_train = y_train.long()
#         self.X_test = X_test.float()
#         self.y_test = y_test.long()
#
#
# #        print(X_train.shape, X_test.shape,
# #              y_train.shape, y_test.shape)
    def get_hidden_act_function(self):
        if self.nn_config_dict["hidden_layer_act"] == "relu":
            return self.relu
        elif self.nn_config_dict["hidden_layer_act"] == "tanh":
            return self.tanh
        elif self.nn_config_dict["hidden_layer_act"] == "sigmoid":
            return self.sigmoid
        else:
            raise ValueError("{} is not a supported hidden layer activation function".format(self.nn_config_dict["hidden_layer_act"]))
       
    def get_final_act_function(self):
        if self.nn_config_dict["final_layer_act"] == "softmax":
            return self.softmax
        else:
            raise ValueError("{} is not a supported hidden layer activation function".format(self.nn_config_dict["final_layer_act"]))
                    
    def forward(self, x):
        hidden = self.fc1(x)
        act = self.hidden_act_func(hidden)
        output = self.fc2(act)
        output = self.softmax(output)
        del hidden, act
        return output

    def set_data(self, df_train_node, train_label, df_test_node, test_label):
        # dataset - load the entire dataset into memory
        # 
        X_train = df_train_node[[col for col in df_train_node.columns if col != 'label']].values
        y_train = train_label.values
        X_test = df_test_node[[col for col in df_test_node.columns if col != 'label']].values
        y_test = test_label.values
        X_train, y_train, X_test, y_test = map(torch.tensor, (X_train, y_train, X_test, y_test))

        self.X_train = X_train.float()
        self.y_train = y_train.long()
        self.X_test = X_test.float()
        self.y_test = y_test.long()



class TwoLayerNeuralNetwork(torch.nn.Module):
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.nn_config_dict = {}
    
    def initialize(self, nn_config_dict):
        self.nn_config_dict = nn_config_dict
        super(TwoLayerNeuralNetwork, self).__init__()
        self.input_size = self.X_train.shape[1]
        self.hidden_size1 = nn_config_dict["numhidden_1"]
        self.hidden_size2 = nn_config_dict["numhidden_2"]
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size1)
        
        # Define the activation functions to be used
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = softmax
        
        # Define hidden layer and final layer activastion functions
        self.hidden_act_func = self.get_hidden_act_function()
        self.final_act_func = self.get_final_act_function()

        self.fc2 = torch.nn.Linear(self.hidden_size1, self.hidden_size2)
        self.fc3 = torch.nn.Linear(self.hidden_size2, 2)
        
    # def load_data(self, dataset, base_dir, feature_split, node_id=None):
    #     if node_id is None:
    #         train_filename = "{}_{}.csv".format(dataset, "train_binary")
    #         test_filename = "{}_{}.csv".format(dataset, "test_binary")
    #
    #     else:
    #         train_filename = "{}_{}_{}.csv".format(dataset, "train", node_id)
    #         test_filename = "{}_{}_{}.csv".format(dataset, "test", node_id)
    #
    #     df_train = pd.read_csv(os.path.join(base_dir,
    #                                   dataset,
    #                                   "feature_split_" + str(feature_split),
    #                                  train_filename), header=None).to_numpy()
    #
    #
    #     df_test = pd.read_csv(os.path.join(base_dir,
    #                                   dataset,
    #                                   "feature_split_" + str(feature_split),
    #                                   test_filename), header=None).to_numpy()
    #
    #
    #     # Split into features and labels
    #     X_train = df_train[:,1:]
    #     y_train = df_train[:,0]
    #
    #     X_test= df_test[:,1:]
    #     y_test = df_test[:,0]
    #
    #
    #     # Map to tensors so torch can use this data
    #     X_train, y_train, X_test, y_test = map(torch.tensor, (X_train, y_train, X_test, y_test))
    #
    #     # Convert features to float and labels to long
    #     self.X_train = X_train.float()
    #     self.y_train = y_train.long()
    #     self.X_test = X_test.float()
    #     self.y_test = y_test.long()
    #
    #
    #     print(X_train.shape, X_test.shape,
    #           y_train.shape, y_test.shape)
        
    def get_hidden_act_function(self):
        if self.nn_config_dict["hidden_layer_act"] == "relu":
            return self.relu
        elif self.nn_config_dict["hidden_layer_act"] == "tanh":
            return self.tanh
        elif self.nn_config_dict["hidden_layer_act"] == "sigmoid":
            return self.sigmoid
        else:
            raise ValueError("{} is not a supported hidden layer activation function".format(self.nn_config_dict["hidden_layer_act"]))
       
    def get_final_act_function(self):
        if self.nn_config_dict["final_layer_act"] == "softmax":
            return self.softmax
        else:
            raise ValueError("{} is not a supported hidden layer activation function".format(self.nn_config_dict["final_layer_act"]))
        
        
    def forward(self, x):
        hidden1 = self.fc1(x)
        act1 = self.hidden_act_func(hidden1)
        hidden2 = self.fc2(act1)
        act2 = self.hidden_act_func(hidden2)
        output = self.fc3(act2)
        output = self.final_act_func(output)
        return output

    def set_data(self, df_train_node, train_label, df_test_node, test_label):
        # dataset - load the entire dataset into memory
        # 
        X_train = df_train_node[[col for col in df_train_node.columns if col != 'label']].values
        y_train = train_label.values
        X_test = df_test_node[[col for col in df_test_node.columns if col != 'label']].values
        y_test = test_label.values
        X_train, y_train, X_test, y_test = map(torch.tensor, (X_train, y_train, X_test, y_test))

        self.X_train = X_train.float()
        self.y_train = y_train.long()
        self.X_test = X_test.float()
        self.y_test = y_test.long()


class ThreeLayerNeuralNetwork(torch.nn.Module):
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.nn_config_dict = {}
    
    def initialize(self, nn_config_dict):
        self.nn_config_dict = nn_config_dict
        super(ThreeLayerNeuralNetwork, self).__init__()
        self.input_size = self.X_train.shape[1]
        self.hidden_size1 = nn_config_dict["numhidden_1"]
        self.hidden_size2 = nn_config_dict["numhidden_2"]
        self.hidden_size3 = nn_config_dict["numhidden_3"]
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size1)
        
        # Define the activation functions to be used
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = softmax
        
        # Define hidden layer and final layer activastion functions
        self.hidden_act_func = self.get_hidden_act_function()
        self.final_act_func = self.get_final_act_function()

        self.fc2 = torch.nn.Linear(self.hidden_size1, self.hidden_size2)
        self.fc3 = torch.nn.Linear(self.hidden_size2, self.hidden_size3)
        self.fc4 = torch.nn.Linear(self.hidden_size2, 2)
        
    # def load_data(self, dataset, base_dir, feature_split, node_id=None):
    #     if node_id is None:
    #         train_filename = "{}_{}.csv".format(dataset, "train_binary")
    #         test_filename = "{}_{}.csv".format(dataset, "test_binary")
    #
    #     else:
    #         train_filename = "{}_{}_{}.csv".format(dataset, "train", node_id)
    #         test_filename = "{}_{}_{}.csv".format(dataset, "test", node_id)
    #
    #     df_train = pd.read_csv(os.path.join(base_dir,
    #                                   dataset,
    #                                   "feature_split_" + str(feature_split),
    #                                  train_filename), header=None).to_numpy()
    #
    #
    #     df_test = pd.read_csv(os.path.join(base_dir,
    #                                   dataset,
    #                                   "feature_split_" + str(feature_split),
    #                                   test_filename), header=None).to_numpy()
    #
    #
    #     # Split into features and labels
    #     X_train = df_train[:,1:]
    #     y_train = df_train[:,0]
    #
    #     X_test= df_test[:,1:]
    #     y_test = df_test[:,0]
    #
    #
    #     # Map to tensors so torch can use this data
    #     X_train, y_train, X_test, y_test = map(torch.tensor, (X_train, y_train, X_test, y_test))
    #
    #     # Convert features to float and labels to long
    #     self.X_train = X_train.float()
    #     self.y_train = y_train.long()
    #     self.X_test = X_test.float()
    #     self.y_test = y_test.long()
    #
    #
    #     print(X_train.shape, X_test.shape,
    #           y_train.shape, y_test.shape)
        
    def get_hidden_act_function(self):
        if self.nn_config_dict["hidden_layer_act"] == "relu":
            return self.relu
        elif self.nn_config_dict["hidden_layer_act"] == "tanh":
            return self.tanh
        elif self.nn_config_dict["hidden_layer_act"] == "sigmoid":
            return self.sigmoid
        else:
            raise ValueError("{} is not a supported hidden layer activation function".format(self.nn_config_dict["hidden_layer_act"]))
       
    def get_final_act_function(self):
        if self.nn_config_dict["final_layer_act"] == "softmax":
            return self.softmax
        else:
            raise ValueError("{} is not a supported hidden layer activation function".format(self.nn_config_dict["final_layer_act"]))
        
        
    def forward(self, x):
        hidden1 = self.fc1(x)
        act1 = self.hidden_act_func(hidden1)
        hidden2 = self.fc2(act1)
        act2 = self.hidden_act_func(hidden2)
        hidden3 = self.fc3(act2)
        act3 = self.hidden_act_func(hidden3)
        output = self.fc4(act3)
        output = self.final_act_func(output)
        return output

    def set_data(self, df_train_node, train_label, df_test_node, test_label):
        # dataset - load the entire dataset into memory
        # 
        X_train = df_train_node[[col for col in df_train_node.columns if col != 'label']].values
        y_train = train_label.values
        X_test = df_test_node[[col for col in df_test_node.columns if col != 'label']].values
        y_test = test_label.values
        X_train, y_train, X_test, y_test = map(torch.tensor, (X_train, y_train, X_test, y_test))

        self.X_train = X_train.float()
        self.y_train = y_train.long()
        self.X_test = X_test.float()
        self.y_test = y_test.long()

def test_cluster():
    nn_cluster = NeuralNetworkCluster()    
    nn_config_0 = {"dataset_name": "arcene",
                   "node_id": 0,
                   "nn_type": "mlp",
                   "num_layers": 2,
                   "loss_function": "cross_entropy",
                   "activation_function": "relu",
                   "learning_rate": 0.1,
                   "feature_split": 1,
                   "run_type": "distributed",
                   "neighbor": 1}
    
    nn_config_1 = {"dataset_name": "arcene",
                   "node_id": 1,
                   "nn_type": "mlp",
                   "num_layers": 2,
                   "loss_function": "cross_entropy",
                   "activation_function": "relu",
                   "learning_rate": 0.1,
                   "feature_split": 1,
                   "run_type": "distributed",
                   "neighbor": 0}
    
    nn_cluster.appendNNToCluster(nn_config_0)
    nn_cluster.appendNNToCluster(nn_config_1)
    
    # Gossip many times
    for i in range(50):
        nn_cluster.gossip(0, 1)
    
    print(nn_cluster.neuralNetDict[0]["train_losses"])
    print(nn_cluster.neuralNetDict[1]["train_losses"])
        
    
    
        
if __name__ == "__main__":
    test_cluster()
#    node_id = 0
#    neighbor_node_id = 1
#    
#    (X_train0, X_test0, y_train0, y_test0) = load_data("arcene", "C:/Users/nitin/eclipse-workspace/consensus-deep-learning-version-2.0/data/", 1, node_id)
#    (X_train1, X_test1, y_train1, y_test1) = load_data("arcene", "C:/Users/nitin/eclipse-workspace/consensus-deep-learning-version-2.0/data/", 1, neighbor_node_id)
#    
#    
#    
#    model0 = Feedforward(X_train0.shape[1], 50)
#    model1 = Feedforward(X_train1.shape[1], 50)
#    
#    criterion0 = torch.nn.CrossEntropyLoss()
#    criterion1 = torch.nn.CrossEntropyLoss()
#    
#    optimizer0 = torch.optim.SGD(model0.parameters(), lr = 0.01)
#    optimizer1 = torch.optim.SGD(model1.parameters(), lr = 0.01)
#       
#    print(model0.eval())
#    print(model1.eval())
#    
#    y_pred0 = model0(X_train0)
#    y_pred1 = model1(X_train1)
#    
#    before_train0 = criterion0(y_pred0.squeeze(), y_train0)
#    before_train1 = criterion1(y_pred1.squeeze(), y_train1)
#    
#
##    y_pred_mean = (y_pred0 + y_pred1)/2
##    
##    y_pred_mean = (y_pred0 + y_pred1)/2
#    
#
#    print('Train loss 0 before training' , before_train0.item()) 
#    print('Train loss 1 before training' , before_train1.item())
#    
#    
#    model0.train()
#    model1.train()
#    epoch = 2000
#    for epoch in range(epoch):
#        optimizer0.zero_grad()
#        optimizer1.zero_grad()
#        # Forward pass
#        y_pred0 = model0(X_train0)
#        y_pred1 = model1(X_train1)
#        
#        
#        y_pred0_2 = y_pred0.clone()
#        y_pred1_2 = y_pred1.clone()
#        
#        y_pred_mean0 = (y_pred0 + y_pred1)/2
#        y_pred_mean1 = (y_pred0_2 + y_pred1_2)/2
#        
#        
#        # Compute Loss
#        loss0 = criterion0(y_pred_mean0.squeeze(), y_train0)
#        loss1 = criterion1(y_pred_mean1.squeeze(), y_train1)
#       
#        # Backward pass
#        loss0.backward(retain_graph=True)
#        loss1.backward(retain_graph=True)
#        optimizer0.step()
#        optimizer1.step()
#        print(loss0.item(), loss1.item())
#        
#        if epoch % 50 == 0:
#            # Print test loss
#            y_pred0 = model0(X_test0)
#            y_pred1 = model1(X_test1)
#            test_loss0 = criterion0(y_pred0.squeeze(), y_test0)
#            test_loss1 = criterion1(y_pred1.squeeze(), y_test1)
#            
#            print("TEST LOSSES at epoch {}: {}, {}".format(epoch, test_loss0, test_loss1))

    