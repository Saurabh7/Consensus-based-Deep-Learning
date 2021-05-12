# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 21:16:28 2020

@author: nitin
"""

from flask import Flask, jsonify
import sys
import flask
import pandas as pd
import json
import traceback
import matplotlib.pyplot as plt
import io
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from scripts.neuralnet import NeuralNetworkCluster
import os
import time
import torch
import gc
import psutil
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
os.environ["CUDA_VISIBLE_DEVICES"]=""
app = Flask(__name__)

# This dictionary will store all the neural networks
base_dir =  "/home/ubuntu/consensus_based_dl/data"
nn_cluster = NeuralNetworkCluster(base_dir)
run_time = 0
start_time = 0
end_time = 0
tensor_count = 0
epoch = 0
save_weights = False


def save_results(op_path):
    """
    Stores a pickle file of the NeuralNetworkCluster object
    """
    global nn_cluster, base_dir
    import pickle
    pickle.dump(nn_cluster, open(os.path.join(op_path, "results.pkl"), "wb"))
    
def shutdown_server():
    func = flask.request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    
@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/plot.png')
def plot_png():
    fig = create_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return 

@app.route("/vpnn/<command>", methods=['GET','POST'])
def updateWPProject(command):
    global nn_cluster, run_time, start_time, end_time, epoch, save_weights
    """project_data = user+++++pw+++++pj_jsondoc.
    This will update an existing project data or insert a non-existant project data item."""
    # print("Executing {} ".format(command))
    if flask.request.method == 'POST':
            nnconfig = flask.request.form['nnconfig']
            
            if nnconfig:
                try:
                    nnconfig_dict = json.loads(nnconfig)
                    # doing a mod here because Peersim does not reset pegasosnode counter
                    nnconfig_dict["node_id"] = nnconfig_dict["node_id"]%nnconfig_dict["num_nodes"]
                    
                    
                    if command == "clear":
                        nn_cluster = NeuralNetworkCluster(base_dir)
                        start_time = time.time()
                        run_time = 0
                        epoch= 0
                        import gc
                        gc.collect()

                    if command == "init":
                        if len(nn_cluster.neuralNetDict) == 0:
                            num_nodes = int(nnconfig_dict["num_nodes"])

                            nn_cluster.init_data(nnconfig_dict)
                        nn_cluster.appendNNToCluster(nnconfig_dict)
                    
                    if command == "train":
                        epoch += 1
                        if epoch % 50 == 0:
                            print("GPU MEMORY ALLOCATED at epoch {}: {}".format(epoch, torch.cuda.memory_allocated()))
                            print("CPU MEMORY USED: {}".format(dict(psutil.virtual_memory()._asdict()))) 
                            print("Clearing tensors and collecting garbage...")
                            import gc
                            gc.collect()
                            torch.cuda.empty_cache()
                        nn_cluster.train(nnconfig_dict["node_id"])
                        #Return the converged flag of this particular neural net
                        return jsonify(nn_cluster.neuralNetDict[nnconfig_dict["node_id"]]["converged_flag"])
                        
                    if command == "calc_losses":
                        epoch += 1
                        nn_cluster.compute_losses_and_accuracies()
                        # Save weights
                        # if save_weights:
                        #     weights_save_dir = os.path.join(base_dir, nnconfig_dict['resourcepath'], 'results', 'weights_{0}'.format(nnconfig_dict['run']))
                        #     if not os.path.exists(weights_save_dir):
                        #         os.makedirs(weights_save_dir)
                        #     nn_cluster.save_linear_layer_weights(weights_save_dir, epoch)
                        
                    if command == "plot":
                        end_time = time.time()
                        run_time = end_time - start_time
#                        if node_id in neural_network_dict.keys():
#                            plot_x, plot_y = list(range(neural_network_dict[node_id].epochs)), neural_network_dict[node_id].loss_arr 
#                            # Check https://stackoverflow.com/questions/50728328/python-how-to-show-matplotlib-in-flask
#                            fig, ax = plt.subplots()
#                            ax.plot(plot_x, plot_y)
#                            output = io.BytesIO()
#                            FigureCanvas(fig).print_png(output)
#                            return Response(output.getvalue(), mimetype='image/png')
                        
                        loss_df = pd.DataFrame(columns=["node", "iter", "train_loss", "test_loss",
                                                        "train_accuracy", "test_accuracy", "converged_state", "overall_train_accuracy", 
                                                        "overall_test_accuracy", "overall_train_auc", 
                                                        "overall_test_auc", "run_time", "converged_flags"])
                        
                        for node_id in nn_cluster.neuralNetDict.keys():
                            
                            train_losses = nn_cluster.neuralNetDict[node_id]["train_losses"]
                            test_losses = nn_cluster.neuralNetDict[node_id]["test_losses"]
                            train_accuracies = nn_cluster.neuralNetDict[node_id]["train_accuracy"]
                            test_accuracies = nn_cluster.neuralNetDict[node_id]["test_accuracy"]
                            converged_states = nn_cluster.neuralNetDict[node_id]["converged_states"]
                            overall_train_accuracies = nn_cluster.neuralNetDict[node_id]["overall_train_accuracy"]
                            overall_test_accuracies = nn_cluster.neuralNetDict[node_id]["overall_test_accuracy"]
                            overall_train_aucs = nn_cluster.neuralNetDict[node_id]["overall_train_auc"]
                            overall_test_aucs = nn_cluster.neuralNetDict[node_id]["overall_test_auc"]
                            converged_flags = nn_cluster.neuralNetDict[node_id]["converged_flags"]
                            fc1_weight = nn_cluster.neuralNetDict[node_id]["fc1_weight"]

                            nodes = [node_id]*len(train_losses)
                            iters = list(range(len(train_losses)))
                            run_times = [run_time] + [None]*(len(train_losses)-1)

                            # Create results dataframe
                            df = pd.DataFrame(data={"node": nodes, "iter": iters, 
                                                    "train_loss": train_losses, "test_loss":test_losses,
                                                    "train_accuracy":train_accuracies, "test_accuracy":test_accuracies,
                                                    "converged_state":converged_states,
                                                    "overall_train_accuracy": overall_train_accuracies, 
                                                    "overall_test_accuracy": overall_test_accuracies,
                                                    "overall_train_auc": overall_train_aucs,
                                                    "overall_test_auc": overall_test_aucs,
                                                    "run_time": run_times,
                                                    "converged_flags": converged_flags,
                                                    "fc1_weight": fc1_weight
                                                    })
                            loss_df = loss_df.append(df)
                            
                        o_path = os.path.join(base_dir, nnconfig_dict['resourcepath'], 'results')
                        if not os.path.exists(o_path):
                            os.makedirs(o_path)
                        
                        f_path = os.path.join(o_path, 'results_{0}.csv'.format(nnconfig_dict['run']))
                        loss_df.to_csv(f_path, index=False)
                        print(nnconfig_dict['run'])
#                        shutdown_server()
                        
                    if command == "gossip":
                        if epoch % 50 == 0:
                            # print("GPU MEMORY ALLOCATED at epoch {}: {}".format(epoch, torch.cuda.memory_allocated() ))
                            # print("CPU MEMORY USED: {}".format(dict(psutil.virtual_memory()._asdict()))) 
                            # print("Clearing tensors and collecting garbage...")
                            import gc
                            gc.collect()
                            # torch.cuda.empty_cache()
                            
                            
#                        start_tensors, start_size = get_tensors_in_memory()
#                        print("Tensors: {}".format(start_tensors))
                        # Gossip only works for runtype == "distributed"
                        neighbor_node_id = nnconfig_dict["neighbor"]%nnconfig_dict["num_nodes"]
                        assert nnconfig_dict["run_type"] == "distributed", "Cannot gossip in centralized setting"

                        # Perform gossip with neighbor's dict as the parameter and update both neural networks
#                        print("Node {} is gossipping with node {}.".format(nnconfig_dict["node_id"] , neighbor_node_id))
                        nn_cluster.gossip( nnconfig_dict["node_id"] , neighbor_node_id, epoch)
#                        end_tensors, end_size = get_tensors_in_memory()
#                        print("Number of tensors added in compute_losses_and_accuracies: {}, size added: {}".format(end_tensors - start_tensors, end_size-start_size))
                        # Return converged_flag of node_id
                        return jsonify(nn_cluster.neuralNetDict[nnconfig_dict["node_id"]]["converged_flag"])
                        
                    
                    if command == "save_results":
                        op_path = os.path.join(base_dir, nnconfig_dict["dataset_name"])
                        save_results(op_path, nnconfig_dict)
#                        print("saved results")
                        
#                    print("Number of neural networks currently existing: {}".format(len(nn_cluster.neuralNetDict.keys())))
                    
                
                except Exception as e:
                    track = traceback.format_exc()
                    print(track)
    
    return jsonify("false")


#
#if __name__ == '__main__':
#    socketio.run(app, debug=False)
