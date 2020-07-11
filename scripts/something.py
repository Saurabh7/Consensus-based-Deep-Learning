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

import torch
import warnings
import os
import logging

import torch.nn.functional as F

warnings.filterwarnings('ignore')


# Todo: data loading should be done in another class
# Todo: Make each model into a class and pass this into higher NNTrainer class

def softmax(t):
    return t.exp() / t.exp().sum(-1).unsqueeze(-1)


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
        self.convergence_epsilon = 0
        self.convergence_iters = 0

    #        self.converged_iters = 0

    def init_data(self,
                  nn_config):  # dataset, num_nodes, feature_split_type, nn_config["random_seed"], overlap_ratio=None):
        #        print(nn_config)
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

        num_nodes = int(nn_config["num_nodes"])
        if nn_config["feature_split_type"] == "random":
            used_indices = []
            num_features = len([col for col in self.df_train.columns if col not in ['label']])
            num_features_split = int(np.ceil(num_features / float(num_nodes)))

            idx_dict = {}
            for split in range(num_nodes):
                # if split == numsplits - 1:
                #   num_features_split = num_features - len(used_indices)

                remaining_indices = [i for i in range(num_features) if i not in used_indices]
                idx = random.sample(remaining_indices, num_features_split)
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
            #            print('Number of features:', len(overlap_features), num_features_split)
            idx_dict = {}
            for split in range(num_nodes):
                # if split == numsplits - 1:
                #   num_features_split = num_features - len(used_indices)

                remaining_indices = [i for i in range(num_features) if i not in used_indices]
                idx = random.sample(remaining_indices, num_features_split)
                idx_dict[split] = idx + overlap_features
                used_indices = used_indices + idx

        self.feature_dict = idx_dict

    #        print(len(self.feature_dict))

    def appendNNToCluster(self, nn_config):
        node_id = nn_config["node_id"]
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
        self.neuralNetDict[node_id]["converged_iters"] = 0
        self.neuralNetDict[node_id]["converged_states"] = []
        self.neuralNetDict[node_id]["converged_flag"] = False
        self.neuralNetDict[node_id]["overall_train_accuracy"] = []
        self.neuralNetDict[node_id]["overall_test_accuracy"] = []
        self.neuralNetDict[node_id]["converged_flag"] = "false"
        self.neuralNetDict[node_id]["converged_flags"] = []

    def gossip(self, node_id, neighbor_node_id):
        """
        Performs gossip on two given node_ids.
        """
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

        y_pred_mean0 = (y_pred0 + y_pred1) / 2
        y_pred_mean1 = (y_pred0_2 + y_pred1_2) / 2

        # Compute Loss
        loss0 = criterion0(y_pred_mean0.squeeze(), model0.y_train)
        loss1 = criterion1(y_pred_mean1.squeeze(), model1.y_train)

        ## If the abs diff between current loss and previous loss < convergence_epsilon
        try:
            diff = abs(loss0.item() - self.neuralNetDict[node_id]["train_losses"][-1])
            if diff < self.convergence_epsilon:
                self.neuralNetDict[node_id]["converged_iters"] += 1
            else:
                self.neuralNetDict[node_id]["converged_iters"] = 0
        except IndexError:
            self.neuralNetDict[node_id]["converged_iters"] = 0

        #        self.neuralNetDict[node_id]["converged_states"].append(self.neuralNetDict[node_id]["converged_iters"])

        # Backward pass
        loss0.backward(retain_graph=True)
        loss1.backward(retain_graph=True)
        optimizer0.step()
        optimizer1.step()

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

        y_pred_train_agg = []
        y_pred_test_agg = []

        for node_id in self.neuralNetDict.keys():
            #            print("Current Nodes" , self.neuralNetDict.keys())

            print("node_id", node_id)
            model = self.neuralNetDict[node_id]["model"]
            criterion = self.neuralNetDict[node_id]["criterion"]

            # Compute Train Loss
            y_pred_train = model(model.X_train)
            #            print(type(y_pred_train))
            y_pred_train = y_pred_train.squeeze()
            train_loss = criterion(y_pred_train, model.y_train)
            train_output = (y_pred_train[:, 1] > 0.5).float()
            train_correct = (train_output == model.y_train).float().sum()
            train_accuracy = train_correct / model.X_train.shape[0]
            self.neuralNetDict[node_id]["train_losses"].append(train_loss.item())
            self.neuralNetDict[node_id]["train_accuracy"].append(train_accuracy.item())

            # Compute Test Loss
            y_pred_test = model(model.X_test)
            y_pred_test = y_pred_test.squeeze()
            test_loss = criterion(y_pred_test, model.y_test)
            test_output = (y_pred_test[:, 1] > 0.5).float()
            test_correct = (test_output == model.y_test).float().sum()
            test_accuracy = test_correct / model.X_test.shape[0]

            self.neuralNetDict[node_id]["test_losses"].append(test_loss.item())
            self.neuralNetDict[node_id]["test_accuracy"].append(test_accuracy.item())
            self.neuralNetDict[node_id]["converged_states"].append(self.neuralNetDict[node_id]["converged_iters"])
            self.neuralNetDict[node_id]["converged_flags"].append(self.neuralNetDict[node_id]["converged_flag"])

            y_pred_train_agg.append(y_pred_train[:, 1].float())
            y_pred_test_agg.append(y_pred_test[:, 1].float())

        # Obtain average predictions
        y_pred_train_agg_pyt = torch.stack(y_pred_train_agg, 0)
        y_pred_train_agg_pyt = torch.mean(y_pred_train_agg_pyt, 0)
        overall_train_output = (y_pred_train_agg_pyt > 0.5).float()
        overall_train_correct = (overall_train_output == model.y_train).float().sum()
        overall_train_accuracy = overall_train_correct / model.X_train.shape[0]

        y_pred_test_agg_pyt = torch.stack(y_pred_test_agg, 0)
        y_pred_test_agg_pyt = torch.mean(y_pred_test_agg_pyt, 0)
        overall_test_output = (y_pred_test_agg_pyt > 0.5).float()
        overall_test_correct = (overall_test_output == model.y_test).float().sum()
        overall_test_accuracy = overall_test_correct / model.X_test.shape[0]

        for node_id in self.neuralNetDict.keys():
            self.neuralNetDict[node_id]["overall_train_accuracy"].append(overall_train_accuracy.item())
            self.neuralNetDict[node_id]["overall_test_accuracy"].append(overall_test_accuracy.item())

    #            iters += [i for i in range(len(self.neuralNetDict[node_id]["test_losses"]))]
    #            nodes += [node_id]*len(self.neuralNetDict[node_id]["test_losses"])
    #            nodes_train_losses += self.neuralNetDict[node_id]["train_losses"]
    #            nodes_train_accuracy += self.neuralNetDict[node_id]["train_accuracy"]
    #            nodes_test_losses += self.neuralNetDict[node_id]["test_losses"]
    #            nodes_test_accuracy += self.neuralNetDict[node_id]["test_accuracy"]
    #            nodes_converged_states += self.neuralNetDict[node_id]["converged_states"]
    #
    #            print(len(iters), len(nodes_train_losses), len(nodes_test_losses), len(nodes_converged_states))
    #
    #        df = pd.DataFrame({'node': nodes, 'iter': iters,
    #                           'train_loss': nodes_train_losses, 'train_accuracy': nodes_train_accuracy,
    #                           'test_loss': nodes_test_losses, 'test_accuracy': nodes_test_accuracy, "converged_iters":nodes_converged_states})
    #        o_path = os.path.join(self.base_dir, self.nn_config['resourcepath'], 'results')
    #        if not os.path.exists(o_path):
    #            os.makedirs(o_path)
    #
    #        f_path = os.path.join(o_path, 'results_{0}.csv'.format(self.nn_config['run']))
    #        df.to_csv(f_path, index=False)
    #        print(self.nn_config['run'])

    # def set_data(df_train_node, df_test_node):
    #     # dataset - load the entire dataset into memory
    #     #
    #     self.X_train = df_train_node[[col for col in df_train_node.columns if col != 'label']].float()
    #     self.y_train = df_train_node['label'].long()
    #     self.X_test = df_test_node[[col for col in df_test_node.columns if col != 'label']].float()
    #     self.y_test = df_test_node['label'].long()

    def train(self, node_id):
        """
        Used for training on only one node in centralized execution.
        No gossip is performed here.
        """

        if self.neuralNetDict[node_id]["converged_iters"] >= self.convergence_iters:
            print("Node {} has converged.".format(node_id))
            #            self.neuralNetDict[node_id]["converged_states"].append(self.neuralNetDict[node_id]["converged_iters"])
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
        try:
            diff = abs(loss0.item() - self.neuralNetDict[node_id]["train_losses"][-1])
            if diff < self.convergence_epsilon:
                self.neuralNetDict[node_id]["converged_iters"] += 1
            else:
                self.neuralNetDict[node_id]["converged_iters"] = 0
        except IndexError:
            self.neuralNetDict[node_id]["converged_iters"] = 0

        #        self.neuralNetDict[node_id]["converged_states"].append(self.neuralNetDict[node_id]["converged_iters"])

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
        self.hidden_size = nn_config_dict["numhidden_1"]
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)

        # Define the activation functions to be used
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = softmax

        self.fc2 = torch.nn.Linear(self.hidden_size, 2)
        # Define hidden layer and final layer activastion functions
        self.hidden_act_func = self.get_hidden_act_function()
        self.final_act_func = self.get_final_act_function()

    def load_data(self, dataset, base_dir, feature_split, node_id=None):
        #         if node_id is None:
        #             train_filename = "{}_{}.csv".format(dataset, "train_binary")
        #             test_filename = "{}_{}.csv".format(dataset, "test_binary")
        #
        #         else:
        #             train_filename = "{}_{}_{}.csv".format(dataset, "train", node_id)
        #             test_filename = "{}_{}_{}.csv".format(dataset, "test", node_id)
        #
        df_train = pd.read_csv(os.path.join(base_dir,
                                            dataset,
                                            "feature_split_" + str(feature_split),
                                            train_filename), header=None).to_numpy()

        df_test = pd.read_csv(os.path.join(base_dir,
                                           dataset,
                                           "feature_split_" + str(feature_split),
                                           test_filename), header=None).to_numpy()

        # Split into features and labels
        X_train = df_train[:, 1:]
        y_train = df_train[:, 0]

        X_test = df_test[:, 1:]
        y_test = df_test[:, 0]

        # Map to tensors so torch can use this data
        X_train, y_train, X_test, y_test = map(torch.tensor, (X_train, y_train, X_test, y_test))

        # Convert features to float and labels to long
        self.X_train = X_train.float()
        self.y_train = y_train.long()
        self.X_test = X_test.float()
        self.y_test = y_test.long()

    #        print(X_train.shape, X_test.shape,
    #              y_train.shape, y_test.shape)
    def get_hidden_act_function(self):
        if self.nn_config_dict["hidden_layer_act"] == "relu":
            return self.relu
        elif self.nn_config_dict["hidden_layer_act"] == "tanh":
            return self.tanh
        elif self.nn_config_dict["hidden_layer_act"] == "sigmoid":
            return self.sigmoid
        else:
            raise ValueError("{} is not a supported hidden layer activation function".format(
                self.nn_config_dict["hidden_layer_act"]))

    def get_final_act_function(self):
        if self.nn_config_dict["final_layer_act"] == "softmax":
            return self.softmax
        else:
            raise ValueError(
                "{} is not a supported hidden layer activation function".format(self.nn_config_dict["final_layer_act"]))

    def forward(self, x):
        hidden = self.fc1(x)
        act = self.hidden_act_func(hidden)
        output = self.fc2(act)
        output = self.softmax(output)
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


#        print(X_train.shape, X_test.shape,
#              y_train.shape, y_test.shape)


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

    def load_data(self, dataset, base_dir, feature_split, node_id=None):
        if node_id is None:
            train_filename = "{}_{}.csv".format(dataset, "train_binary")
            test_filename = "{}_{}.csv".format(dataset, "test_binary")

        else:
            train_filename = "{}_{}_{}.csv".format(dataset, "train", node_id)
            test_filename = "{}_{}_{}.csv".format(dataset, "test", node_id)

        df_train = pd.read_csv(os.path.join(base_dir,
                                            dataset,
                                            "feature_split_" + str(feature_split),
                                            train_filename), header=None).to_numpy()

        df_test = pd.read_csv(os.path.join(base_dir,
                                           dataset,
                                           "feature_split_" + str(feature_split),
                                           test_filename), header=None).to_numpy()

        # Split into features and labels
        X_train = df_train[:, 1:]
        y_train = df_train[:, 0]

        X_test = df_test[:, 1:]
        y_test = df_test[:, 0]

        # Map to tensors so torch can use this data
        X_train, y_train, X_test, y_test = map(torch.tensor, (X_train, y_train, X_test, y_test))

        # Convert features to float and labels to long
        self.X_train = X_train.float()
        self.y_train = y_train.long()
        self.X_test = X_test.float()
        self.y_test = y_test.long()

        print(X_train.shape, X_test.shape,
              y_train.shape, y_test.shape)

    def get_hidden_act_function(self):
        if self.nn_config_dict["hidden_layer_act"] == "relu":
            return self.relu
        elif self.nn_config_dict["hidden_layer_act"] == "tanh":
            return self.tanh
        elif self.nn_config_dict["hidden_layer_act"] == "sigmoid":
            return self.sigmoid
        else:
            raise ValueError("{} is not a supported hidden layer activation function".format(
                self.nn_config_dict["hidden_layer_act"]))

    def get_final_act_function(self):
        if self.nn_config_dict["final_layer_act"] == "softmax":
            return self.softmax
        else:
            raise ValueError(
                "{} is not a supported hidden layer activation function".format(self.nn_config_dict["final_layer_act"]))

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


#        print(X_train.shape, X_test.shape,
#              y_train.shape, y_test.shape)


def test_cluster():
    nn_cluster = NeuralNetworkCluster()
    nn_config_0 = {"dataset_name": "arcene",
                   "node_id": 0,
                   "nn_type": "mlp",
                   "num_layers": 2,
                   "loss_function": "cross_entropy",
                   "activation_function": "relu",
                   "learning_rate": 0.1,
