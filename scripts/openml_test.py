"""
========
Datasets
========

A basic tutorial on how to list, load and visualize datasets.
"""
############################################################################
# In general, we recommend working with tasks, so that the results can
# be easily reproduced. Furthermore, the results can be compared to existing results
# at OpenML. However, for the purposes of this tutorial, we are going to work with
# the datasets directly.

# License: BSD 3-Clause

import openml
import numpy as np
import os		
import pandas as pd
############################################################################
# List datasets
# =============

# def get_dataset(id, save_path):
# 	dataset = openml.datasets.get_dataset(id)

# 	# Print a summary
# 	print(f"This is dataset '{dataset.name}', the target feature is "
# 	      f"'{dataset.default_target_attribute}'")
# 	print(f"URL: {dataset.url}")
# 	print(dataset.description[:500])

# 	############################################################################
# 	# Load a dataset
# 	# ==============

# 	# X - An array/dataframe where each row represents one example with
# 	# the corresponding feature values.
# 	# y - the classes for each example
# 	# categorical_indicator - an array that indicates which feature is categorical
# 	# attribute_names - the names of the features for the examples (X) and
# 	# target feature (y)
# 	X, y, categorical_indicator, attribute_names = dataset.get_data(
# 		include_row_id=True,
# 	    dataset_format='dataframe',
# 	    target=dataset.default_target_attribute
# 	)
# 	# X = np.hstack((X, y))
# 	X = np.nan_to_num(X)
# 	y = y.reshape(-1, 1)
# 	X = np.hstack((y, X))
# 	np.savetxt(save_path, X, delimiter=',')



# # Get desxter dataset
# get_dataset(4136, "../data/dexter/dexter_data.csv")
# get_dataset(4137, "../data/dorothea/dorothea_data.csv")

def load_nips_dataset(data_dir, dataset_name):
	import scipy.io as sio
	mat_fname = "../data/MatlabData/" + dataset_name + ".mat"
	mat_contents = sio.loadmat(mat_fname)
	try:
		X_train = mat_contents['X_train'].todense()
	except AttributeError:
		X_train = mat_contents['X_train']
	y_train = mat_contents['Y_train'].squeeze().reshape(-1,1)
	
	try:
		X_valid = mat_contents['X_valid'].todense()
	except AttributeError:
		X_valid = mat_contents['X_valid']


	y_valid = mat_contents['Y_valid'].squeeze().reshape(-1,1)

	train_data = np.hstack((y_train, X_train))
	validation_data = np.hstack((y_valid, X_valid))

	#np.savetxt(os.path.join(data_dir, dataset_name + "_train.csv"), train_data, delimiter=',')
	#np.savetxt(os.path.join(data_dir, dataset_name + "_valid.csv"), validation_data, delimiter=',')
	pd.DataFrame(data=train_data, columns=['f'+str(i) for i in range(train_data.shape[1])]).to_csv(os.path.join(data_dir, dataset_name + "_train.csv"), index=False, header=False)
	pd.DataFrame(data=validation_data, columns=['f'+str(i) for i in range(validation_data.shape[1])]).to_csv(os.path.join(data_dir, dataset_name + "_valid.csv"), index=False, header=False)

def main():
	data_dir = "../data"
	print("Saving dorothea")
	load_nips_dataset(data_dir, "dorothea")

	print("Saving dexter")
	load_nips_dataset(data_dir, "dexter")

	print("Saving arcene")
	load_nips_dataset(data_dir, "arcene")

	print("Saving madelon")
	load_nips_dataset(data_dir, "madelon")

	# print("Saving gisette")
	# load_nips_dataset(data_dir, "gisette")


main()