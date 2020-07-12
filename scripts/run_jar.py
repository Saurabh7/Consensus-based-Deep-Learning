# -*- coding: utf-8 -*-
"""
Created on Thu May 28 17:06:45 2020

@author: nitin
"""

# Run jar

import subprocess
import os

base_config_path ="../config/arcene/"
config_files = os.listdir(base_config_path)


config_ids = set([r.split(".")[0].split("_")[1] for r in config_files])

from subprocess import Popen as new
from time import sleep

results_path = "../data/mnist_balanced/results"
result_files = os.listdir(results_path)
completed_result_ids = set([r.split(".")[0].split("_")[1] for r in result_files])

print("Completed: {}".format(completed_result_ids))

runnable_config_ids = config_ids.difference(completed_result_ids)

config_paths = [os.path.join(base_config_path, f) for f in config_files]
for cid in runnable_config_ids:
    cfile = "config_{}.cfg".format(cid)
    cpath = os.path.join(base_config_path, cfile)
    if os.path.exists(cpath):
        print("Running ", cfile)
#        proc1 = new(["python", "flask_server.py"], shell=True)
#        sleep(7)
#        proc2 = new(["java", "-jar", "C:\\Users\\nitin\\eclipse-workspace\\consensus-deep-learning-version-2.0\\consensus2.jar",
#                    p], shell=True)
#        sleep(7)
#        proc1.terminate()
#        proc2.terminate()
        subprocess.run(["java", "-jar", "C:\\Users\\nitin\\eclipse-workspace\\consensus_based_dl_2.0\\consensus2.jar",
                    cpath])
    

    