# Vertically partition datasets

import pandas as pd
filename = "../dl4j-examples/dl4j-examples/target/classes/classification/mnist_train.csv"
df = pd.read_csv(filename)
print(df.head())