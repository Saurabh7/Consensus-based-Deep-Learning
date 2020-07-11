import pandas as pd
import os
import numpy as np
from copy import copy
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import logging
import random
import math



# constants
DATA_DIR = "../data/"


def load_args():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("-n", "--numsplits", type=int, default=10,
	                    help="Number of vertical splits to make on the data.")
	
	parser.add_argument("-d", "--dataset", type=str, 
	                    help="Name of dataset")
	parser.add_argument("-r", "--run", type=int, default=0,
	                    help="Run number, determines random seed.")
	
	args = parser.parse_args()
	return args



def loadData(DATA_DIR, filename):
    """
    Loads csv files where column 0 represents labels.
    """
    df = pd.read_csv(os.path.join(DATA_DIR, filename), header=None)
    df2 = np.array(df)
    labels = df2[:,0]
    df2 = df2[:,1:]
    return df2, labels


def makeBinary(data, labels, classLabel):
    """
    Makes the dataset binary by changing the given class label to 1. 
    """
    zippedData = list(zip(data, labels))
    zippedData = [[dataPoint, 1] if label == 8 else [dataPoint, 0] for dataPoint, label in zippedData]
    data, labels = list(zip(*zippedData))
    data, labels = np.array(data), np.array(labels)
    assert data.shape[0] == labels.shape[0]
    print(data.shape, labels.shape)
    return data, labels

def makeBinaryBalanced(data, labels, posClassLabel, negClassLabel):
    """
    Makes the dataset binary and balanced by changing the posClassLabel class to 0, and the negClassLabel class to 1.
    """
    zippedData = list(zip(data, labels))
    zippedData = [[dataPoint, 1] if label == posClassLabel else [dataPoint, 0] for dataPoint, label in zippedData 
                  if label in [posClassLabel, negClassLabel]]
    
    data, labels = list(zip(*zippedData))
    data, labels = np.array(data), np.array(labels)
    assert data.shape[0] == labels.shape[0]
    print(data.shape, labels.shape)
    return data, labels


def split_padded(matrix,labels, n):
    a = np.arange(matrix.shape[1])
    padding = (-len(a))%n
    index_arrays = np.split(np.concatenate((a,np.zeros(padding))).astype(int),n)
    index_arrays[-1] = np.trim_zeros(index_arrays[-1] , "b")
    matrices = [np.hstack((labels[:,np.newaxis], matrix[:, index_arrays[i]])) for i in range(len(index_arrays))]
    return matrices


def verticalPartition(data, labels):
    assert round(sum(probVector),3) == 1
    df = copy(data)
    numFeatures = data.shape[1]
    splitDfs = []
    for i in range(len(probVector)):
        numFeats = probVector[i]*numFeatures
        tempDf = df[:,0:min(int(numFeats), df.shape[1]-1)]
        tempDf = np.hstack((labels[:,np.newaxis], tempDf))
        splitDfs.append(tempDf)
        df = df[:, int(numFeats)+1:]
    return splitDfs


def saveSplitFiles(DATA_DIR, baseFilename, splitDfs):
    """
    Saves files into respective CSV files.
    """
    for i in range(len(splitDfs)):
        temp_df = splitDfs[i]
        temp_filename = baseFilename.split(".")[0] + "_" + str(i) + ".csv"        
        temp_df = pd.DataFrame(data=temp_df, index=None)
        temp_df.to_csv(os.path.join(DATA_DIR, temp_filename), index=False, header=False)
        print("File saved in {}".format(os.path.join(DATA_DIR, temp_filename)))


def random_feature_split(seed, X_t, X_v):
	# Change feature splits
	X_train, X_valid = np.copy(X_t), np.copy(X_v) # Make a copy so we don't modify the original's reference
	np.random.seed(seed)
	train_set_length = X_train.shape[0]
	X = np.vstack((X_train, X_valid))
	X = np.transpose(X)
	np.random.shuffle(X)
	X = np.transpose(X)
	# Split back into train and valid
	X_train, X_valid = X[:train_set_length,:], X[train_set_length:,:]
	return X_train, X_valid

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def split_random(df_train, df_test, numsplits):
	used_indices = []
	
	df_train_splits = []
	df_test_splits = []

	labels_train = df_train[df_train.columns[0]]
	labels_test = df_test[df_test.columns[0]]

	df_train = df_train.iloc[:, 1:]
	df_test = df_test.iloc[:, 1:]


	num_features = df_train.shape[1]
	num_features_split = int(np.ceil(num_features / float(numsplits)))

	idx_dict = {}
	for split in range(numsplits):
		# if split == numsplits - 1:
		# 	num_features_split = num_features - len(used_indices) 

		remaining_indices = [i for i in range(num_features) if i not in used_indices]
		idx = random.sample(remaining_indices, num_features_split)
		idx_dict[split] = idx
		used_indices = used_indices + idx
		df_train_split = df_train.iloc[:,idx]
		df_train_split.insert(loc=0, column='0', value = labels_train)

		df_test_split = df_test.iloc[:,idx]
		df_test_split.insert(loc=0, column='0', value = labels_test)

		df_train_splits.append(df_train_split)
		df_test_splits.append(df_test_split)

	return df_train_splits, df_test_splits, idx_dict


# def split_random(df_train, df_test, numsplits):
#     used_indices = []
#     num_features = df_train.shape[1]
#     num_features_split = math.ceil(num_features / float(numsplits))
#     print(num_features_split)
#     df_train_splits = []
#     df_test_splits = []

#     labels_train = df_train[df_train.columns[0]]
#     labels_test = df_test[df_test.columns[0]]

#     idx_dict = {}
#     for split in range(numsplits):
#         if split == numsplits - 1:
#             num_features_split = num_features - len(used_indices) - 1

#         remaining_indices = [i for i in range(1, num_features) if i not in used_indices]
#         idx = random.sample(remaining_indices, num_features_split)
#         print(len(idx))
#         idx_dict[split] = idx
#         used_indices = used_indices + idx
#         df_train_split = df_train.iloc[:,idx]
#         df_train_split.insert(loc=0, column='0', value = labels_train)

#         df_test_split = df_test.iloc[:,idx]
#         df_test_split.insert(loc=0, column='0', value = labels_test)

#         df_train_splits.append(df_train_split)
#         df_test_splits.append(df_test_split)

#     return df_train_splits, df_test_splits, idx_dict


def process_sonar(num_splits, run):

	dataset_dir = os.path.join(DATA_DIR, "sonar")
	trainFilename = "sonar_train.csv"
	testFilename = "sonar_test.csv"

	# Process training set
	data, labels = loadData(dataset_dir, trainFilename)
	print(data.shape, labels.shape)
	splitDfs = split_padded(data, labels, num_splits)
	print([a.shape for a in splitDfs])
	saveSplitFiles(dataset_dir, trainFilename, splitDfs)
	print(Counter(labels))
	df_to_save = pd.DataFrame(np.hstack((labels.reshape(-1,1), data)))
	print(df_to_save.shape)
	df_to_save.to_csv(os.path.join(dataset_dir, trainFilename.split(".")[0] + "_binary.csv"), index=False, header=False)



	# Process test set
	data, labels = loadData(dataset_dir, testFilename)
	splitDfs = split_padded(data, labels, num_splits)
	print([a.shape for a in splitDfs])
	saveSplitFiles(dataset_dir, testFilename, splitDfs)
	print(Counter(labels))

	df_to_save = pd.DataFrame(np.hstack((labels.reshape(-1,1), data)))
	print(df_to_save.shape)
	df_to_save.to_csv(os.path.join(dataset_dir, testFilename.split(".")[0] + "_binary.csv"), index=False, header=False)

	print("Sonar data available in {}".format(dataset_dir))


def process_arcene(num_splits, run):

	feature_splits = 5
	main_data_dir = os.path.join(DATA_DIR, "arcene")
	for i in range(feature_splits):
		dataset_dir = os.path.join(DATA_DIR, "arcene", "feature_split_{}".format(i+1))
		if not os.path.exists(dataset_dir):
			os.makedirs(dataset_dir)

		trainDataFilename = "arcene_train.csv"
		validDataFilename = "arcene_valid.csv"
		arcene_train_data = pd.read_csv(os.path.join(main_data_dir, trainDataFilename), header=None)
		arcene_valid_data = pd.read_csv(os.path.join(main_data_dir, validDataFilename), header=None)
		
		X_arcene_train, y_arcene_train = arcene_train_data.iloc[:, 1:], arcene_train_data.iloc[:, 0]
		X_arcene_train = np.array(X_arcene_train)
		y_arcene_train = np.array(y_arcene_train) 
		y_arcene_train[y_arcene_train == -1] = 0
		y_arcene_train[y_arcene_train == 1] = 1

		X_arcene_valid, y_arcene_valid = arcene_valid_data.iloc[:, 1:], arcene_valid_data.iloc[:, 0]
		X_arcene_valid = np.array(X_arcene_valid)
		y_arcene_valid = np.array(y_arcene_valid) 
		y_arcene_valid[y_arcene_valid == -1] = 0
		y_arcene_valid[y_arcene_valid == 1] = 1


		# Normalize data
		scaler = MinMaxScaler().fit(X_arcene_train)
		X_arcene_train_orig = scaler.transform(X_arcene_train)
		X_arcene_valid_orig = scaler.transform(X_arcene_valid)

		X_arcene_train, X_arcene_valid = random_feature_split(i*12345, X_arcene_train_orig, X_arcene_valid_orig)


		# Hstack labels
		all_train = np.hstack((y_arcene_train.reshape(-1,1), X_arcene_train_orig))
		all_test = np.hstack((y_arcene_valid.reshape(-1,1), X_arcene_valid_orig))

		
		pd.DataFrame(all_train).to_csv(os.path.join(dataset_dir, "arcene_train_binary.csv"), index=None, header=None)
		pd.DataFrame(all_test).to_csv(os.path.join(dataset_dir, "arcene_test_binary.csv"), index=None, header=None)

		
		trainFilename = "arcene_train.csv"
		testFilename = "arcene_test.csv"

		splitTrainDfs, splitValidDfs, idx_dict = split_random(pd.DataFrame(all_train), pd.DataFrame(all_test), num_splits)
		print([a.shape for a in splitTrainDfs])
		print([a.shape for a in splitValidDfs])

		saveSplitFiles(dataset_dir, trainFilename, splitTrainDfs)
		saveSplitFiles(dataset_dir, testFilename, splitValidDfs)
		import pickle
		pickle.dump(idx_dict, open(os.path.join(dataset_dir, "split_feature_indices.pkl"), "wb"))

		print("Arcene data available in {}".format(dataset_dir))


def process_mnist(num_splits, run):

	feature_splits = 3

	def split_random(df_train, df_test, numsplits):
		used_indices = []
		num_features = df_train.shape[1]
		num_features_split = int(num_features / float(numsplits))

		df_train_splits = []
		df_test_splits = []

		idx_dict = {}
		for split in range(numsplits):
			if split == numsplits - 1:
				num_features_split = num_features - len(used_indices) - 1

			remaining_indices = [i for i in range(1, num_features) if i not in used_indices]
			idx = random.sample(remaining_indices, num_features_split)
			idx_dict[split] = idx
			df_train_split = df_train.iloc[:,idx]
			df_test_split = df_test.iloc[:,idx]

			df_train_splits.append(df_train_split)
			df_test_splits.append(df_test_split)

		return df_train_splits, df_test_splits, idx_dict


	main_data_dir = os.path.join(DATA_DIR, "mnist")
	for i in range(feature_splits):
		dataset_dir = os.path.join(DATA_DIR, "mnist", "feature_split_{}".format(i+1))
		if not os.path.exists(dataset_dir):
			os.makedirs(dataset_dir)

		trainDataFilename = "mnist_train_binary.csv"
		validDataFilename = "mnist_test_binary.csv"
		arcene_train_data = pd.read_csv(os.path.join(main_data_dir, trainDataFilename), header=None)
		arcene_valid_data = pd.read_csv(os.path.join(main_data_dir, validDataFilename), header=None)
		
		X_arcene_train, y_arcene_train = arcene_train_data.iloc[:, 1:], arcene_train_data.iloc[:, 0]
		X_arcene_train = np.array(X_arcene_train)
		y_arcene_train = np.array(y_arcene_train) 
		y_arcene_train[y_arcene_train == -1] = 0
		y_arcene_train[y_arcene_train == 1] = 1

		X_arcene_valid, y_arcene_valid = arcene_valid_data.iloc[:, 1:], arcene_valid_data.iloc[:, 0]
		X_arcene_valid = np.array(X_arcene_valid)
		y_arcene_valid = np.array(y_arcene_valid) 
		y_arcene_valid[y_arcene_valid == -1] = 0
		y_arcene_valid[y_arcene_valid == 1] = 1


		# Normalize data
		scaler = MinMaxScaler().fit(X_arcene_train)
		X_arcene_train_orig = scaler.transform(X_arcene_train)
		X_arcene_valid_orig = scaler.transform(X_arcene_valid)

		X_arcene_train, X_arcene_valid = random_feature_split(i*12345, X_arcene_train_orig, X_arcene_valid_orig)


		# Hstack labels
		all_train = np.hstack((y_arcene_train.reshape(-1,1), X_arcene_train_orig))
		all_test = np.hstack((y_arcene_valid.reshape(-1,1), X_arcene_valid_orig))

		
		pd.DataFrame(all_train).to_csv(os.path.join(dataset_dir, "mnist_train_binary.csv"), index=None, header=None)
		pd.DataFrame(all_test).to_csv(os.path.join(dataset_dir, "mnist_test_binary.csv"), index=None, header=None)

		
		trainFilename = "mnist_train.csv"
		testFilename = "mnist_test.csv"

		splitTrainDfs, splitValidDfs, idx_dict = split_random(pd.DataFrame(all_train), pd.DataFrame(all_test), num_splits)
		print([a.shape for a in splitTrainDfs])
		print([a.shape for a in splitValidDfs])

		saveSplitFiles(dataset_dir, trainFilename, splitTrainDfs)
		saveSplitFiles(dataset_dir, testFilename, splitValidDfs)
		import pickle
		pickle.dump(idx_dict, open(os.path.join(dataset_dir, "split_feature_indices.pkl"), "wb"))

		print("Mnist data available in {}".format(dataset_dir))


def process_dexter(num_splits, run):

	feature_splits = 5
	main_data_dir = os.path.join(DATA_DIR, "dexter")
	for i in range(feature_splits):
		dataset_dir = os.path.join(DATA_DIR, "dexter", "feature_split_{}".format(i+1))
		if not os.path.exists(dataset_dir):
			os.makedirs(dataset_dir)

		trainDataFilename = "dexter_train.csv"
		validDataFilename = "dexter_valid.csv"
		dexter_train_data = pd.read_csv(os.path.join(main_data_dir, trainDataFilename), header=None)
		dexter_valid_data = pd.read_csv(os.path.join(main_data_dir, validDataFilename), header=None)
		
		X_dexter_train, y_dexter_train = dexter_train_data.iloc[:, 1:], dexter_train_data.iloc[:, 0]
		X_dexter_train = np.array(X_dexter_train)
		y_dexter_train = np.array(y_dexter_train) 
		y_dexter_train[y_dexter_train == -1] = 0
		y_dexter_train[y_dexter_train == 1] = 1

		X_dexter_valid, y_dexter_valid = dexter_valid_data.iloc[:, 1:], dexter_valid_data.iloc[:, 0]
		X_dexter_valid = np.array(X_dexter_valid)
		y_dexter_valid = np.array(y_dexter_valid) 
		y_dexter_valid[y_dexter_valid == -1] = 0
		y_dexter_valid[y_dexter_valid == 1] = 1


		# Normalize data
		scaler = MinMaxScaler().fit(X_dexter_train)
		X_dexter_train_orig = scaler.transform(X_dexter_train)
		X_dexter_valid_orig = scaler.transform(X_dexter_valid)

		X_dexter_train, X_dexter_valid = random_feature_split(i*12345, X_dexter_train_orig, X_dexter_valid_orig)

		# Hstack labels
		all_train = np.hstack((y_dexter_train.reshape(-1,1), X_dexter_train_orig))
		all_test = np.hstack((y_dexter_valid.reshape(-1,1), X_dexter_valid_orig))

		
		pd.DataFrame(all_train).to_csv(os.path.join(dataset_dir, "dexter_train_binary.csv"), index=None, header=None)
		pd.DataFrame(all_test).to_csv(os.path.join(dataset_dir, "dexter_test_binary.csv"), index=None, header=None)

		
		trainFilename = "dexter_train.csv"
		testFilename = "dexter_test.csv"

		# Process training set
		splitDfs = split_padded(X_dexter_train, y_dexter_train, num_splits)
		print([a.shape for a in splitDfs])
		saveSplitFiles(dataset_dir, trainFilename, splitDfs)
		print(Counter(y_dexter_train))


		# Process valid set
		splitDfs = split_padded(X_dexter_valid, y_dexter_valid, num_splits)
		print([a.shape for a in splitDfs])
		saveSplitFiles(dataset_dir, testFilename, splitDfs)
		print(Counter(y_dexter_valid))

		print("dexter data available in {}".format(dataset_dir))





def process_dorothea(num_splits, run):

	feature_splits = 1
	main_data_dir = os.path.join(DATA_DIR, "dorothea")
	for i in range(feature_splits):
		dataset_dir = os.path.join(DATA_DIR, "dorothea", "feature_split_{}".format(i+1))
		if not os.path.exists(dataset_dir):
			os.makedirs(dataset_dir)

		trainDataFilename = "dorothea_train.csv"
		validDataFilename = "dorothea_valid.csv"
		dorothea_train_data = pd.read_csv(os.path.join(main_data_dir, trainDataFilename), header=None)
		dorothea_valid_data = pd.read_csv(os.path.join(main_data_dir, validDataFilename), header=None)
		
		X_dorothea_train, y_dorothea_train = dorothea_train_data.iloc[:, 1:], dorothea_train_data.iloc[:, 0]
		X_dorothea_train = np.array(X_dorothea_train)
		y_dorothea_train = np.array(y_dorothea_train) 
		y_dorothea_train[y_dorothea_train == -1] = 0
		y_dorothea_train[y_dorothea_train == 1] = 1

		X_dorothea_valid, y_dorothea_valid = dorothea_valid_data.iloc[:, 1:], dorothea_valid_data.iloc[:, 0]
		X_dorothea_valid = np.array(X_dorothea_valid)
		y_dorothea_valid = np.array(y_dorothea_valid) 
		y_dorothea_valid[y_dorothea_valid == -1] = 0
		y_dorothea_valid[y_dorothea_valid == 1] = 1


		# Normalize data
		scaler = MinMaxScaler().fit(X_dorothea_train)
		X_dorothea_train_orig = scaler.transform(X_dorothea_train)
		X_dorothea_valid_orig = scaler.transform(X_dorothea_valid)

		X_dorothea_train, X_dorothea_valid = random_feature_split(i*12345, X_dorothea_train_orig, X_dorothea_valid_orig)

		# Hstack labels
		all_train = np.hstack((y_dorothea_train.reshape(-1,1), X_dorothea_train_orig))
		all_test = np.hstack((y_dorothea_valid.reshape(-1,1), X_dorothea_valid_orig))

		
		pd.DataFrame(all_train).to_csv(os.path.join(dataset_dir, "dorothea_train_binary.csv"), index=None, header=None)
		pd.DataFrame(all_test).to_csv(os.path.join(dataset_dir, "dorothea_test_binary.csv"), index=None, header=None)

		
		trainFilename = "dorothea_train.csv"
		testFilename = "dorothea_test.csv"

		# # Process training set
		# splitDfs = split_padded(X_dorothea_train, y_dorothea_train, num_splits)
		# print([a.shape for a in splitDfs])
		# saveSplitFiles(dataset_dir, trainFilename, splitDfs)
		# print(Counter(y_dorothea_train))


		# # Process valid set
		# splitDfs = split_padded(X_dorothea_valid, y_dorothea_valid, num_splits)
		# print([a.shape for a in splitDfs])
		# saveSplitFiles(dataset_dir, testFilename, splitDfs)
		# print(Counter(y_dorothea_valid))

		splitTrainDfs, splitValidDfs, idx_dict = split_random(pd.DataFrame(all_train), pd.DataFrame(all_test), num_splits)
		print([a.shape for a in splitTrainDfs])
		print([a.shape for a in splitValidDfs])

		saveSplitFiles(dataset_dir, trainFilename, splitTrainDfs)
		saveSplitFiles(dataset_dir, testFilename, splitValidDfs)
		import pickle
		pickle.dump(idx_dict, open(os.path.join(dataset_dir, "split_feature_indices.pkl"), "wb"))

		print("dorothea data available in {}".format(dataset_dir))


def process_madelon(num_splits, run):

	feature_splits = 5

	main_data_dir = os.path.join(DATA_DIR, "madelon")
	for i in range(5):

		dataset_dir = os.path.join(DATA_DIR, "madelon", "feature_split_{}".format(i+1))
		if not os.path.exists(dataset_dir):
			os.makedirs(dataset_dir)

		trainDataFilename = "madelon_train.csv"
		validDataFilename = "madelon_valid.csv"
		madelon_train_data = pd.read_csv(os.path.join(main_data_dir, trainDataFilename), header=None)
		madelon_valid_data = pd.read_csv(os.path.join(main_data_dir, validDataFilename), header=None)
		
		X_madelon_train, y_madelon_train = madelon_train_data.iloc[:, 1:], madelon_train_data.iloc[:, 0]
		X_madelon_train = np.array(X_madelon_train)
		y_madelon_train = np.array(y_madelon_train) 
		y_madelon_train[y_madelon_train == -1] = 0
		y_madelon_train[y_madelon_train == 1] = 1

		X_madelon_valid, y_madelon_valid = madelon_valid_data.iloc[:, 1:], madelon_valid_data.iloc[:, 0]
		X_madelon_valid = np.array(X_madelon_valid)
		y_madelon_valid = np.array(y_madelon_valid) 
		y_madelon_valid[y_madelon_valid == -1] = 0
		y_madelon_valid[y_madelon_valid == 1] = 1


		# Normalize data
		scaler = MinMaxScaler().fit(X_madelon_train)
		X_madelon_train_orig = scaler.transform(X_madelon_train)
		X_madelon_valid_orig = scaler.transform(X_madelon_valid)


		X_madelon_train, X_madelon_valid = random_feature_split(i*12345, X_madelon_train_orig, X_madelon_valid_orig)
		# Hstack labels
		all_train = np.hstack((y_madelon_train.reshape(-1,1), X_madelon_train_orig))
		all_test = np.hstack((y_madelon_valid.reshape(-1,1), X_madelon_valid_orig))

		
		pd.DataFrame(all_train).to_csv(os.path.join(dataset_dir, "madelon_train_binary.csv"), index=None, header=None)
		pd.DataFrame(all_test).to_csv(os.path.join(dataset_dir, "madelon_test_binary.csv"), index=None, header=None)
		
		trainFilename = "madelon_train.csv"
		testFilename = "madelon_test.csv"

		# Process training set
		splitDfs = split_padded(X_madelon_train, y_madelon_train, num_splits)
		print([a.shape for a in splitDfs])
		saveSplitFiles(dataset_dir, trainFilename, splitDfs)
		print(Counter(y_madelon_train))

		# Process test set
		splitDfs = split_padded(X_madelon_valid, y_madelon_valid, num_splits)
		print([a.shape for a in splitDfs])
		saveSplitFiles(dataset_dir, testFilename, splitDfs)
		print(Counter(y_madelon_valid))

		print("madelon data available in {}".format(dataset_dir))


def process_ccat(num_splits, run):
	from sklearn.datasets import load_svmlight_file

	feature_splits = 5

	main_data_dir = os.path.join(DATA_DIR, "ccat")
	for i in range(5):
		dataset_dir = os.path.join(DATA_DIR, "ccat", "feature_split_{}".format(i+1))
		if not os.path.exists(dataset_dir):
			os.makedirs(dataset_dir)
		trainDataFilename = "ccat.trn"
		validDataFilename = "ccat.tst"
		ccat_train_data, ccat_train_labels = load_svmlight_file(os.path.join(main_data_dir, trainDataFilename))
		ccat_valid_data, ccat_valid_labels = load_svmlight_file(os.path.join(main_data_dir, validDataFilename))
		print(ccat_train_data.todense().shape, ccat_valid_data.todense().shape)


def process_adult(num_splits, run):
	from sklearn.datasets import load_svmlight_file

	feature_splits = 5
	main_data_dir = os.path.join(DATA_DIR, "adult")
	trainDataFilename = "adult.trn"
	validDataFilename = "adult.tst"
	X_adult_train, y_adult_train = load_svmlight_file(os.path.join(main_data_dir, trainDataFilename))
	X_adult_valid, y_adult_valid = load_svmlight_file(os.path.join(main_data_dir, validDataFilename))
	
	X_adult_train_orig = X_adult_train.todense()
	X_adult_valid_orig = X_adult_valid.todense()

	y_adult_train[y_adult_train == -1] = 0
	y_adult_train[y_adult_train == 1] = 1

	y_adult_valid[y_adult_valid == -1] = 0
	y_adult_valid[y_adult_valid == 1] = 1

	
	for i in range(5):
		dataset_dir = os.path.join(DATA_DIR, "adult", "feature_split_{}".format(i+1))
		if not os.path.exists(dataset_dir):
			os.makedirs(dataset_dir)
		


		# Hstack labels
		all_train = np.hstack((y_adult_train.reshape(-1,1), X_adult_train_orig))
		all_test = np.hstack((y_adult_valid.reshape(-1,1), X_adult_valid_orig))

		X_adult_train, X_adult_valid = random_feature_split(i*12345, X_adult_train_orig, X_adult_valid_orig)

		pd.DataFrame(all_train).to_csv(os.path.join(dataset_dir, "adult_train_binary.csv"), index=None, header=None)
		pd.DataFrame(all_test).to_csv(os.path.join(dataset_dir, "adult_test_binary.csv"), index=None, header=None)

		print("Saving binary files")
		print(X_adult_train.shape, X_adult_valid.shape)

		trainFilename = "adult_train.csv"
		testFilename = "adult_test.csv"

		# Process training set
		splitDfs = split_padded(X_adult_train, y_adult_train, num_splits)
		print([a.shape for a in splitDfs])
		saveSplitFiles(dataset_dir, trainFilename, splitDfs)
		print(Counter(y_adult_train))


		# Process test set
		splitDfs = split_padded(X_adult_valid, y_adult_valid, num_splits)
		print([a.shape for a in splitDfs])
		saveSplitFiles(dataset_dir, testFilename, splitDfs)
		print(Counter(y_adult_valid))
		print("Cov2 data available in {}".format(dataset_dir))
		

def process_covtype(num_splits, nitinnat):
	from sklearn.datasets import load_svmlight_file

	feature_splits = 5
	main_data_dir = os.path.join(DATA_DIR, "cov2")
	trainDataFilename = "covtype_bn_train.trn"
	validDataFilename = "covtype_bn_test.tst"
	X_cov2_train, y_cov2_train = load_svmlight_file(os.path.join(main_data_dir, trainDataFilename))
	X_cov2_valid, y_cov2_valid = load_svmlight_file(os.path.join(main_data_dir, validDataFilename))
	
	X_cov2_train_orig = X_cov2_train.todense()
	X_cov2_valid_orig = X_cov2_valid.todense()

	y_cov2_train[y_cov2_train == -1] = 0
	y_cov2_train[y_cov2_train == 1] = 1

	y_cov2_valid[y_cov2_valid == -1] = 0
	y_cov2_valid[y_cov2_valid == 1] = 1

	
	for i in range(5):
		dataset_dir = os.path.join(DATA_DIR, "cov2", "feature_split_{}".format(i+1))
		if not os.path.exists(dataset_dir):
			os.makedirs(dataset_dir)
		


		# Hstack labels
		all_train = np.hstack((y_cov2_train.reshape(-1,1), X_cov2_train_orig))
		all_test = np.hstack((y_cov2_valid.reshape(-1,1), X_cov2_valid_orig))

		X_cov2_train, X_cov2_valid = random_feature_split(i*12345, X_cov2_train_orig, X_cov2_valid_orig)

		pd.DataFrame(all_train).to_csv(os.path.join(dataset_dir, "cov2_train_binary.csv"), index=None, header=None)
		pd.DataFrame(all_test).to_csv(os.path.join(dataset_dir, "cov2_test_binary.csv"), index=None, header=None)

		print("Saving binary files")
		print(X_cov2_train.shape, X_cov2_valid.shape)

		trainFilename = "cov2_train.csv"
		testFilename = "cov2_test.csv"

		# Process training set
		splitDfs = split_padded(X_cov2_train, y_cov2_train, num_splits)
		print([a.shape for a in splitDfs])
		saveSplitFiles(dataset_dir, trainFilename, splitDfs)
		print(Counter(y_cov2_train))


		# Process test set
		splitDfs = split_padded(X_cov2_valid, y_cov2_valid, num_splits)
		print([a.shape for a in splitDfs])
		saveSplitFiles(dataset_dir, testFilename, splitDfs)
		print(Counter(y_cov2_valid))
		print("Cov2 data available in {}".format(dataset_dir))

def process_gisette(num_splits, run):

	feature_splits = 5

	main_data_dir = os.path.join(DATA_DIR, "gisette")
	for i in range(5):
		dataset_dir = os.path.join(DATA_DIR, "gisette", "feature_split_{}".format(i+1))
		if not os.path.exists(dataset_dir):
			os.makedirs(dataset_dir)
		trainDataFilename = "gisette_train.csv"
		validDataFilename = "gisette_valid.csv"
		gisette_train_data = pd.read_csv(os.path.join(main_data_dir, trainDataFilename), header=None)
		gisette_valid_data = pd.read_csv(os.path.join(main_data_dir, validDataFilename), header=None)
		
		X_gisette_train, y_gisette_train = gisette_train_data.iloc[:, 1:], gisette_train_data.iloc[:, 0]
		X_gisette_train = np.array(X_gisette_train)
		y_gisette_train = np.array(y_gisette_train) 
		y_gisette_train[y_gisette_train == -1] = 0
		y_gisette_train[y_gisette_train == 1] = 1

		X_gisette_valid, y_gisette_valid = gisette_valid_data.iloc[:, 1:], gisette_valid_data.iloc[:, 0]
		X_gisette_valid = np.array(X_gisette_valid)
		y_gisette_valid = np.array(y_gisette_valid) 
		y_gisette_valid[y_gisette_valid == -1] = 0
		y_gisette_valid[y_gisette_valid == 1] = 1

		# Normalize data
		scaler = MinMaxScaler().fit(X_gisette_train)
		X_gisette_train_orig = scaler.transform(X_gisette_train)
		X_gisette_valid_orig = scaler.transform(X_gisette_valid)

		# Hstack labels
		all_train = np.hstack((y_gisette_train.reshape(-1,1), X_gisette_train_orig))
		all_test = np.hstack((y_gisette_valid.reshape(-1,1), X_gisette_valid_orig))

		X_gisette_train, X_gisette_valid = random_feature_split(i*12345, X_gisette_train_orig, X_gisette_valid_orig)


		pd.DataFrame(all_train).to_csv(os.path.join(dataset_dir, "gisette_train_binary.csv"), index=None, header=None)
		pd.DataFrame(all_test).to_csv(os.path.join(dataset_dir, "gisette_test_binary.csv"), index=None, header=None)

		print("Saving binary files")
		print(X_gisette_train.shape, X_gisette_valid.shape)

		trainFilename = "gisette_train.csv"
		testFilename = "gisette_test.csv"

		# Process training set
		splitDfs = split_padded(X_gisette_train, y_gisette_train, num_splits)
		print([a.shape for a in splitDfs])
		saveSplitFiles(dataset_dir, trainFilename, splitDfs)
		print(Counter(y_gisette_train))


		# Process test set
		splitDfs = split_padded(X_gisette_valid, y_gisette_valid, num_splits)
		print([a.shape for a in splitDfs])
		saveSplitFiles(dataset_dir, testFilename, splitDfs)
		print(Counter(y_gisette_valid))
		print("Gisette data available in {}".format(dataset_dir))

if __name__ == "__main__":
	args = load_args()
	#TODO: Delete existing csvs first
	
	if args.dataset == "sonar":
		process_sonar(args.numsplits, args.run)

	elif args.dataset == "madelon":
		process_madelon(args.numsplits, args.run)

	elif args.dataset == "gisette":
		process_gisette(args.numsplits, args.run)

	elif args.dataset == "arcene":
		process_arcene(args.numsplits, args.run)

	elif args.dataset == "dorothea":
		process_dorothea(args.numsplits, args.run)

	elif args.dataset == "dexter":
		process_dexter(args.numsplits, args.run)

	elif args.dataset == "ccat":
		process_ccat(args.numsplits, args.run)

	elif args.dataset == "adult":
		process_adult(args.numsplits, args.run)

	elif args.dataset == "covtype":
		process_covtype(args.numsplits, args.run)

	elif args.dataset == "mnist":
		process_mnist(args.numsplits, args.run)