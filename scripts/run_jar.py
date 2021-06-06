# -*- coding: utf-8 -*-
"""
Created on Thu May 28 17:06:45 2020

@author: nitin
"""

# Run jar

import subprocess
import os

for dataset in ['gisette']:#['arcene', 'dexter', 'gisette', 'madelon', 'HT', 'mnist_balanced', 'dorothea_balanced']:
	base_config_path ="../config/{}/".format(dataset)
	config_files = os.listdir(base_config_path)

	print(config_files)
	config_files = [r for r in config_files if r not in ['.DS_Store', 'September', 'old', '2000iter', 'april']]
	print(config_files)
	config_ids = set([r.split(".")[0].split("_")[1] for r in config_files])

	from subprocess import Popen as new
	from time import sleep

	results_path = "../data/{}/results".format(dataset)
	result_files = os.listdir(results_path)
	completed_result_ids = set([])
	result_files = [r for r in result_files if r not in ['.DS_Store', 'September', 'old', '2000iter', 'april']]

	if len(result_files) > 0:
		completed_result_ids = set([r.split(".")[0].split("_")[1] for r in result_files])

	print("Completed: {}".format(completed_result_ids))

	runnable_config_ids = [cid for cid in config_ids if int(cid) in [i for i in range(1,16)]]
	runnable_config_ids = set(runnable_config_ids).difference(completed_result_ids)

	config_paths = [os.path.join(base_config_path, f) for f in config_files]
	for cid in runnable_config_ids:
	    cfile = "config_{}.cfg".format(cid)
	    cpath = os.path.join(base_config_path, cfile)
	    print(cpath)
	    if os.path.exists(cpath):
	        print("Running ", cfile)
	#        proc1 = new(["python", "flask_server.py"], shell=True)
	#        sleep(7)
	#        proc2 = new(["java", "-jar", "C:\\Users\\nitin\\eclipse-workspace\\consensus-deep-learning-version-2.0\\consensus2.jar",
	#                    p], shell=True)
	#        sleep(7)
	#        proc1.terminate()
	#        proc2.terminate()
	        subprocess.run(["java", "-jar", os.environ['ConsensusDLPath'] + "/consensus2.jar",
	                    cpath])
    

    
