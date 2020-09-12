import torch
from scripts.utils.metrics import softmax
import torch.nn.functional as F
import numpy as np

class SingleLayerCNN(torch.nn.Module):
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.nn_config_dict = {}

    def initialize(self, nn_config_dict):
        self.nn_config_dict = nn_config_dict
        super(SingleLayerCNN, self).__init__()
        self.input_size = self.X_train.shape[1]
        
        self.hidden_size  = nn_config_dict["numhidden_1"]
        self.num_filters = 20# int(160 / int(nn_config_dict["num_nodes"]))
        
        kernel_size=3

        self.conv1 = torch.nn.Conv2d(1, 12, stride=1, kernel_size=2)
        self.conv2 = torch.nn.Conv2d(12, 24, stride=1, kernel_size=2)

        self.fc1 = torch.nn.Linear(24, 36)
        self.fc2 = torch.nn.Linear(36, 10)

        # Define the activation functions to be used
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = softmax
        # self.batchnorm1 = torch.nn.BatchNorm1d(self.hidden_size)
        self.dropout = torch.nn.Dropout(p=0.1)

        # self.fc2 = torch.nn.Linear(self.hidden_size, 10)
        # Define hidden layer and final layer activastion functions
        self.hidden_act_func = self.get_hidden_act_function()
        self.final_act_func = self.get_final_act_function()

    def get_hidden_act_function(self):
        if self.nn_config_dict["hidden_layer_act"] == "relu":
            return self.relu
        elif self.nn_config_dict["hidden_layer_act"] == "tanh":
            return self.tanh
        elif self.nn_config_dict["hidden_layer_act"] == "sigmoid":
            return self.sigmoid
        else:
            raise ValueError("{} is not a supported hidden layer activation function".format
                (self.nn_config_dict["hidden_layer_act"]))

    def get_final_act_function(self):
        if self.nn_config_dict["final_layer_act"] == "softmax":
            return self.softmax
        else:
            raise ValueError \
                ("{} is not a supported hidden layer activation function".format(self.nn_config_dict["final_layer_act"]))

    def forward(self, x):
        conv1 = self.conv1(x.view(-1,1,7,7))
        x = self.hidden_act_func(conv1)
        x = F.max_pool2d(x, 2)
#         x = self.dropout1(x)
        conv2 = self.conv2(x)
        x = self.hidden_act_func(conv2)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        # x = self.dropout(x)
        hidden = self.fc1(x)
        act = self.hidden_act_func(hidden)
        output = self.fc2(act)
        # print('Output:', output.shape)
        output = self.softmax(output)
        del hidden
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

    def get_hidden_act_function(self):
        if self.nn_config_dict["hidden_layer_act"] == "relu":
            return self.relu
        elif self.nn_config_dict["hidden_layer_act"] == "tanh":
            return self.tanh
        elif self.nn_config_dict["hidden_layer_act"] == "sigmoid":
            return self.sigmoid
        else:
            raise ValueError("{} is not a supported hidden layer activation function".format
                (self.nn_config_dict["hidden_layer_act"]))

    def get_final_act_function(self):
        if self.nn_config_dict["final_layer_act"] == "softmax":
            return self.softmax
        else:
            raise ValueError \
                ("{} is not a supported hidden layer activation function".format(self.nn_config_dict["final_layer_act"]))

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
        self.dropout = torch.nn.Dropout(p=0.2)

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

    def get_hidden_act_function(self):
        if self.nn_config_dict["hidden_layer_act"] == "relu":
            return self.relu
        elif self.nn_config_dict["hidden_layer_act"] == "tanh":
            return self.tanh
        elif self.nn_config_dict["hidden_layer_act"] == "sigmoid":
            return self.sigmoid
        else:
            raise ValueError("{} is not a supported hidden layer activation function".format
                (self.nn_config_dict["hidden_layer_act"]))

    def get_final_act_function(self):
        if self.nn_config_dict["final_layer_act"] == "softmax":
            return self.softmax
        else:
            raise ValueError \
                ("{} is not a supported hidden layer activation function".format(self.nn_config_dict["final_layer_act"]))

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

    def get_hidden_act_function(self):
        if self.nn_config_dict["hidden_layer_act"] == "relu":
            return self.relu
        elif self.nn_config_dict["hidden_layer_act"] == "tanh":
            return self.tanh
        elif self.nn_config_dict["hidden_layer_act"] == "sigmoid":
            return self.sigmoid
        else:
            raise ValueError("{} is not a supported hidden layer activation function".format
                (self.nn_config_dict["hidden_layer_act"]))

    def get_final_act_function(self):
        if self.nn_config_dict["final_layer_act"] == "softmax":
            return self.softmax
        else:
            raise ValueError \
                ("{} is not a supported hidden layer activation function".format(self.nn_config_dict["final_layer_act"]))

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


