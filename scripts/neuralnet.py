# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 20:46:56 2020
@author: nitin
"""

# Create non-linearly separable data
import numpy as np
import random
import pandas as pd
import torch
import warnings
import os
import logging
from scripts.utils.metrics import roc_auc_compute_fn
from scripts.models import SingleLayerNeuralNetwork, TwoLayerNeuralNetwork, ThreeLayerNeuralNetwork,FourLayerNeuralNetwork, SingleLayerCNN
import pickle

warnings.filterwarnings('ignore')
base_dir = os.environ['ConsensusDLPath'] + "/data"

# Todo: data loading should be done in another class
# Todo: Make each model into a class and pass this into higher NNTrainer class


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
        
        self.df_train = pd.read_csv(os.path.join(self.base_dir, nn_config["dataset_name"], train_filename))
        print('Overall Shape:', self.df_train.shape)
        self.n_classes = self.df_train.label.nunique()

        self.labels_train = self.df_train.pop('label')

        print('Post pop Shape:', self.df_train.shape)
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
                if not remaining_indices:
                    idx_dict[split] = idx_dict[0]
                    continue
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

            # total_feats = int(num_features / (9*(1-nn_config["overlap_ratio"]) + 1))


            
            num_features_overlap = int(np.ceil(nn_config["overlap_ratio"] * num_features))
            num_features_split = int(np.ceil((num_features - num_features_overlap) / float(num_nodes)))

            
            print('Number of total features: {}, Number of features overlap: {}, Number of split features {}'.format(
                    num_features, num_features_overlap, num_features_split
                ))

            overlap_features = random.sample([i for i in range(num_features)], num_features_overlap)
            used_indices += overlap_features
            for split in range(num_nodes):
                # if split == numsplits - 1:
                #   num_features_split = num_features - len(used_indices)

                remaining_indices = [i for i in range(num_features) if i not in used_indices]
                try:
                    idx = random.sample(remaining_indices, num_features_split)
                except:
                    idx = remaining_indices
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

        if nn_config["nn_type"] == "cnn" and nn_config["num_layers"] == 1:
            model = SingleLayerCNN()
            df_train_node = self.df_train.iloc[:, self.feature_dict[node_id]]
            df_test_node = self.df_test.iloc[:, self.feature_dict[node_id]]
            model.set_data(df_train_node, self.labels_train, df_test_node, self.labels_test)
            model.initialize(nn_config)            

        elif nn_config["num_layers"] == 1:
            model = SingleLayerNeuralNetwork(n_classes = self.n_classes)
            df_train_node = self.df_train.iloc[:, self.feature_dict[node_id]]
            df_test_node = self.df_test.iloc[:, self.feature_dict[node_id]]
            model.set_data(df_train_node, self.labels_train, df_test_node, self.labels_test)
            model.initialize(nn_config)

        elif nn_config["num_layers"] == 2:
            model = TwoLayerNeuralNetwork(n_classes = self.n_classes)
            df_train_node = self.df_train.iloc[:, self.feature_dict[node_id]]
            df_test_node = self.df_test.iloc[:, self.feature_dict[node_id]]
            model.set_data(df_train_node, self.labels_train, df_test_node, self.labels_test)
            model.initialize(nn_config)

        elif nn_config["num_layers"] == 3:
            model = ThreeLayerNeuralNetwork(n_classes = self.n_classes)
            df_train_node = self.df_train.iloc[:, self.feature_dict[node_id]]
            df_test_node = self.df_test.iloc[:, self.feature_dict[node_id]]
            model.set_data(df_train_node, self.labels_train, df_test_node, self.labels_test)
            model.initialize(nn_config)
        else:
            model = FourLayerNeuralNetwork(n_classes = self.n_classes)
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
        # optimizer = torch.optim.Adam(model.parameters(), lr=nn_config["learning_rate"], weight_decay=0.000001)
        optimizer = torch.optim.SGD(model.parameters(), lr=nn_config["learning_rate"])
        self.neuralNetDict[node_id]["optimizer"] = optimizer
        
        self.neuralNetDict[node_id]["fc1_weight"] = []
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
        
    def gossip(self, node_id, neighbor_node_id, epoch):
        """
        Performs gossip on two given node_ids.
        """
        # print("Inside gossip")
#        start_tensors, start_size = get_tensors_in_memory()
        # if self.neuralNetDict[node_id]["converged_iters"] >= self.convergence_iters:
        #     print("Node {} has converged.".format(node_id))
        #     self.neuralNetDict[node_id]["converged_flag"] = "true"
        #     return
        # print("node_id", node_id)
        model0 = self.neuralNetDict[node_id]["model"]
        model1 = self.neuralNetDict[neighbor_node_id]["model"]
        
        criterion0 = self.neuralNetDict[node_id]["criterion"]
        criterion1 = self.neuralNetDict[neighbor_node_id]["criterion"]
        
        optimizer0 = self.neuralNetDict[node_id]["optimizer"]
        optimizer1 = self.neuralNetDict[neighbor_node_id]["optimizer"]
        
        # SGD
        # Wipe gradients of both optimizers
        # optimizer0.zero_grad()
        # optimizer1.zero_grad()
        
        # # Forward pass
        # y_pred0 = model0(model0.X_train)

        # y_pred1 = model1(model1.X_train)
        
        # y_pred0_2 = y_pred0.clone()
        # y_pred1_2 = y_pred1.clone()
        
        # y_pred_mean0 = (y_pred0 + y_pred1)/2
        # y_pred_mean1 = (y_pred0_2 + y_pred1_2)/2
        # # print('SHAPES', y_pred_mean0.shape, y_pred_mean1.shape)
        # # Compute Loss
        # loss0 = torch.nn.CrossEntropyLoss(reduction='none')(y_pred_mean0.squeeze(), model0.y_train)
        # loss1 = torch.nn.CrossEntropyLoss(reduction='none')(y_pred_mean1.squeeze(), model1.y_train)
        # # print(loss0.shape, loss1.shape)

        # # for idx, row in enumerate(model0.X_train):
        # optimizer0.zero_grad()
        # optimizer1.zero_grad()
        # # Forward pass
        # # y_pred0 = model0(row.reshape(-1, model0.X_train.shape[1]))        
        # # Compute Loss
        

        # # loss0.backward()
        # # optimizer0.step()
        # # loss0.backward(retain_graph=True)
        # # loss1.backward(retain_graph=True)

        # loss0.backward(torch.ones_like(loss0),retain_graph=True)
        # loss1.backward(torch.ones_like(loss1),retain_graph=True)
        
        # optimizer0.step()
        # optimizer1.step()

        # SGD
        # for idx in range(0,model0.X_train.shape[0], 64):


        # Local Update
        random_sample = random.randint(0, model0.X_train.shape[0])
        optimizer0.zero_grad()
        optimizer1.zero_grad()
    
        y_pred0sample = model0.forward(model0.X_train[random_sample:random_sample+1])
        y_pred1sample = model1.forward(model1.X_train[random_sample:random_sample+1])

        loss0sample = torch.nn.CrossEntropyLoss()(y_pred0sample, model0.y_train[random_sample:random_sample+1])
        loss1sample = torch.nn.CrossEntropyLoss()(y_pred1sample, model1.y_train[random_sample:random_sample+1])

        loss0sample.backward()
        loss1sample.backward()

        optimizer0.step()
        optimizer1.step()

        optimizer0.zero_grad()
        optimizer1.zero_grad()
 

        # Gossip Update
        random_sample2= random.randint(0, model0.X_train.shape[0])

        y_pred0 = model0.forward(model0.X_train[random_sample2:random_sample2+1])
        y_pred1 = model1.forward(model1.X_train[random_sample2:random_sample2+1])

        y_pred0_2 = y_pred0.clone()
        y_pred1_2 = y_pred1.clone()
        y_pred_mean0 = (y_pred0 + y_pred1)/2
        y_pred_mean1 = (y_pred0_2 + y_pred1_2)/2

        loss0 = torch.nn.CrossEntropyLoss()(y_pred_mean0, model0.y_train[random_sample2:random_sample2+1])
        loss1 = torch.nn.CrossEntropyLoss()(y_pred_mean1, model1.y_train[random_sample2:random_sample2+1])

        loss0.backward(retain_graph=True)
        loss1.backward(retain_graph=True)
        
        optimizer0.step()
        optimizer1.step()



        # MINI BATCH
        # for idx in range(0,model0.X_train.shape[0], 64):
        #     optimizer0.zero_grad()
        #     optimizer1.zero_grad()
        
        #     y_pred0 = model0(model0.X_train[idx:idx+64])
        #     y_pred1 = model1(model1.X_train[idx:idx+64])
        #     y_pred0_2 = y_pred0.clone()
        #     y_pred1_2 = y_pred1.clone()
        #     y_pred_mean0 = (y_pred0 + y_pred1)/2
        #     y_pred_mean1 = (y_pred0_2 + y_pred1_2)/2

        #     loss0 = torch.nn.CrossEntropyLoss(reduction='mean')(y_pred_mean0.squeeze(), model0.y_train[idx:idx+64])
        #     loss1 = torch.nn.CrossEntropyLoss(reduction='mean')(y_pred_mean1.squeeze(), model1.y_train[idx:idx+64])

        #     loss0.backward(retain_graph=True)
        #     loss1.backward(retain_graph=True)
            
        #     optimizer0.step()
        #     optimizer1.step()


        # loss0 = criterion0(y_pred_mean0.squeeze(), model0.y_train)
        # loss1 = criterion1(y_pred_mean1.squeeze(), model1.y_train)


        # temp = criterion0(y_pred0.squeeze(), model0.y_train)
        # print('TEMP', temp.item())
        # test_auc_score = roc_auc_compute_fn(y_pred_mean0[:, 1], model0.y_test) 
        # ## If the abs diff between current loss and previous loss < convergence_epsilon
        # if self.neuralNetDict[node_id]["prev_loss"] is None:
        #     self.neuralNetDict[node_id]["converged_iters"] = 0
        # else:
        #     diff = 100
        #     try:
        #         # self.neuralNetDict[node_id]["test_losses"][-1] - loss0.item()# 
        #         diff = test_auc_score - np.min(self.neuralNetDict[node_id]["test_auc"]) 
        #     except:
        #         pass
        #     print(self.convergence_epsilon, node_id, diff, self.neuralNetDict[node_id]["converged_iters"], diff < self.convergence_epsilon)
        #     if diff < self.convergence_epsilon:# and len(self.neuralNetDict[node_id]["test_losses"]) > self.convergence_iters*2:
        #         self.neuralNetDict[node_id]["converged_iters"] += 1
                
        #     else:
        #         self.neuralNetDict[node_id]["converged_iters"] = 0
        # self.neuralNetDict[node_id]["prev_loss"] = loss0.item()
        # Backward pass
        

        # y_pred_test = model0(model0.X_test)
        # y_true_test = model0.y_test

        # test_loss = criterion0(y_pred_test.squeeze(), y_true_test)
        # diff = 100
        # try:
        #     diff = np.min(np.nan_to_num(self.neuralNetDict[node_id]["test_losses"])) - test_loss.item()
        # except:
        #     pass

        # if diff < self.convergence_epsilon:
        #     self.neuralNetDict[node_id]["converged_iters"] += 1
        # else:
        #     self.neuralNetDict[node_id]["converged_iters"] = 0

        # print(epoch, diff, self.neuralNetDict[node_id]["converged_iters"])

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

        pass
    
    def compute_losses_and_accuracies(self):
        """
        Computes train and test losses for all the nodes.
        """
        # print("Inside compute_losses_and_accuracies")
#        start_tensors, start_size = get_tensors_in_memory()
        
        y_pred_train_agg = []
        y_pred_test_agg= []
        # print("Calculating losses: ")
        losses = []
        for node_id in self.neuralNetDict.keys():
            model = self.neuralNetDict[node_id]["model"]
            criterion = self.neuralNetDict[node_id]["criterion"]        
            
            # Compute Train Loss
            y_pred_train = model(model.X_train)
#            print(type(y_pred_train))
            y_pred_train = y_pred_train.squeeze()
            train_loss = criterion(y_pred_train, model.y_train) 
            train_output = np.argmax(y_pred_train.detach().numpy(), axis=1)#.float()#[:, 1]>0.5).float()
            # print('loss', train_loss.item())
            # print('Y Pred TRAIN', y_pred_train.shape, train_output.shape)
            losses.append(train_loss.item())
            train_correct = np.equal(train_output, model.y_train).sum()
            train_accuracy = train_correct/model.X_train.shape[0]
            
            if self.n_classes <= 2:

                train_auc_score = roc_auc_compute_fn(y_pred_train[:, 1].detach(), model.y_train)
            else:
                train_auc_score = train_accuracy

            self.neuralNetDict[node_id]["train_losses"].append(train_loss.item())
            self.neuralNetDict[node_id]["train_accuracy"].append(train_accuracy.item())
            self.neuralNetDict[node_id]["train_auc"].append(train_auc_score.item())
            # print('Shape:', model.fc1.weight.shape)
            self.neuralNetDict[node_id]["fc1_weight"].append(torch.square(model.fc1.weight).sum().item())
            # print(node_id, self.neuralNetDict[node_id]["fc1_weight"][-1])
            # Compute Test Loss
            y_pred_test = model(model.X_test)
            y_pred_test = y_pred_test.squeeze()
            test_loss = criterion(y_pred_test, model.y_test)
            test_output = np.argmax(y_pred_test.detach().numpy(), axis=1)

            test_correct = np.equal(test_output, model.y_test).sum()
            test_accuracy = test_correct/model.X_test.shape[0]
            #losses.append(test_loss.item())

            if self.n_classes <= 2:
                test_auc_score = roc_auc_compute_fn(y_pred_test[:, 1].detach(), model.y_test) 
            else:
                test_auc_score = test_accuracy

            self.neuralNetDict[node_id]["test_losses"].append(test_loss.item())
            self.neuralNetDict[node_id]["test_accuracy"].append(test_accuracy.item())
            self.neuralNetDict[node_id]["test_auc"].append(test_auc_score.item())
            
            self.neuralNetDict[node_id]["converged_states"].append(self.neuralNetDict[node_id]["converged_iters"])
            self.neuralNetDict[node_id]["converged_flags"].append(self.neuralNetDict[node_id]["converged_flag"])
            

            y_pred_train_agg.append(y_pred_train.detach().numpy())
            y_pred_test_agg.append(y_pred_test.detach().numpy())
            
#            del y_pred_train, train_loss, train_output, train_correct, train_accuracy, train_auc_score
#            del y_pred_test, test_loss, test_output, test_correct, test_accuracy, test_auc_score
        
        # Obtain average predictions
        y_pred_train_agg_pyt = np.sum(y_pred_train_agg, axis=0) / len(y_pred_train_agg)
        # print('SHAPE of average', y_pred_train_agg_pyt.shape)

        # y_pred_train_agg_pyt = torch.stack(y_pred_train_agg, 0)
        # y_pred_train_agg_pyt = torch.mean(y_pred_train_agg_pyt, 0)
        overall_train_output = np.argmax(y_pred_train_agg_pyt, axis=1)#(y_pred_train_agg_pyt>0.5).float()
        overall_train_correct = np.equal(overall_train_output, model.y_train).sum()
        overall_train_accuracy = overall_train_correct/model.X_train.shape[0]
        if self.n_classes <= 2:
            overall_train_auc = roc_auc_compute_fn(y_pred_train_agg_pyt[:,1], model.y_train)
        else:
            overall_train_auc = overall_train_accuracy

        y_pred_test_agg_pyt = np.sum(y_pred_test_agg, axis=0) / len(y_pred_test_agg) #torch.stack(y_pred_test_agg, 0)
        # y_pred_test_agg_pyt = torch.mean(y_pred_test_agg_pyt, 0)
        overall_test_output = np.argmax(y_pred_test_agg_pyt, axis=1)#>0.5).float()
        overall_test_correct = np.equal(overall_test_output, model.y_test).sum()
        overall_test_accuracy = overall_test_correct/model.X_test.shape[0]

        if self.n_classes <= 2:
            overall_test_auc = roc_auc_compute_fn(y_pred_test_agg_pyt[:,1], model.y_test)
        else:
            overall_test_auc = overall_test_accuracy
        
        print("Overall Train AUC: {}, Overall Test AUC: {}, Mean Test Loss: {}".format(
            overall_train_auc.item(), overall_test_auc.item(), np.mean(losses)))
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
        
        # if self.neuralNetDict[node_id]["converged_iters"] >= self.convergence_iters:
        #     print("Node {} has converged.".format(node_id))
        #     self.neuralNetDict[node_id]["converged_flag"] = "true"
        #     return

        print("node_id", node_id)
        print("Centralized training")
        model0 = self.neuralNetDict[node_id]["model"]
        criterion0 = self.neuralNetDict[node_id]["criterion"]        
        optimizer0 = self.neuralNetDict[node_id]["optimizer"]        
        
        # SGD
        # y_pred0 = model0(model0.X_train)
        # #
        # #
        # loss0 = torch.nn.CrossEntropyLoss(reduction='none')(y_pred0.squeeze(), model0.y_train)
        # #
        # loss0.backward(torch.ones_like(loss0),retain_graph=True)
        # optimizer0.step()

        idx = random.randint(0, model0.X_train.shape[0])
        optimizer0.zero_grad()
        y_pred0 = model0.forward(model0.X_train[idx:idx+1])    
        loss0 = torch.nn.CrossEntropyLoss()(y_pred0, model0.y_train[idx:idx+1])
        loss0.backward()
        optimizer0.step()



        # BATCH
        # for idx in range(0,model0.X_train.shape[0], 64):
        #     optimizer0.zero_grad()
        
        #     y_pred0 = model0(model0.X_train[idx:idx+64])        
        #     loss0 = torch.nn.CrossEntropyLoss(reduction='mean')(y_pred0.squeeze(), model0.y_train[idx:idx+64])
        #     loss0.backward()
        #     optimizer0.step()

        # If the abs diff between current loss and previous loss < convergence_epsilon
        # if self.neuralNetDict[node_id]["prev_loss"] is None:
        #     self.neuralNetDict[node_id]["converged_iters"] = 0
        # else:
        #     diff = abs(loss0.item() - self.neuralNetDict[node_id]["prev_loss"])
        #     if diff < self.convergence_epsilon:
        #         self.neuralNetDict[node_id]["converged_iters"] += 1
                
        #     else:
        #         self.neuralNetDict[node_id]["converged_iters"] = 0
        # self.neuralNetDict[node_id]["prev_loss"] = loss0.item()

        # Backward pass
        #loss0.backward()#retain_graph=True)
        # Update parameters
        #optimizer0.step()

        # y_pred_test = model0(model0.X_test)
        # y_true_test = model0.y_test

        # test_loss = criterion0(y_pred_test.squeeze(), y_true_test)
        # diff = 100
        # try:
        #     diff = np.min(np.nan_to_num(self.neuralNetDict[node_id]["test_losses"])) - test_loss.item()
        # except:
        #     pass

        # if diff < self.convergence_epsilon:
        #     self.neuralNetDict[node_id]["converged_iters"] += 1
        # else:
        #     self.neuralNetDict[node_id]["converged_iters"] = 0
        
        # print(diff, diff < self.convergence_epsilon, self.neuralNetDict[node_id]["converged_iters"])

    def save_linear_layer_weights(self, weights_save_dir, iter):
        """
        Saves weights of torch.nn.Linear layers as numpy arrays.
        :param model:
        :return:
        """
        weights_dict = {}
        save_path = os.path.join(weights_save_dir, "weights_{}.pkl".format(iter))
        for key in self.neuralNetDict.keys():
            model = self.neuralNetDict[key]["model"]
            linear_layers = [module for module in model.modules() if type(module) == torch.nn.Linear]
            shapes = []
            weights_list = []
            for layer in linear_layers:
                layer_weights = layer.weight.detach().numpy()
                shapes.append(layer_weights.shape)
                weights_list.append(layer_weights)
            weights_dict[key] = weights_list
        pickle.dump(weights_dict, open(save_path, "wb"))



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
