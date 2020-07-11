import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
# final_iv, IV = data_vars(irisi, irisi.target)

importance = mutual_info_classif(mnist_binary, mnist_binary[0], discrete_features=True)

importance_df = pd.DataFrame({'feature_index': [i for i in range(len(importance))], 'importance': importance})

imp_rank = importance_df['importance'].rank(ascending=False)

df = pd.read_csv("mnist_fi/feature_split_1/indices.csv", header=None)
df2 = pd.read_csv("mnist_fi/feature_split_2/indices.csv", header=None)
df3 = pd.read_csv("mnist_fi/feature_split_3/indices.csv", header=None)


node = []

for i in range(9):
    node = node + [i]*79

node = node + [i+1]*(784-len(node))
node


df.columns = ['feature_index']
impdf = df.merge(importance_df, on=['feature_index'], how='inner')
impdf['Node'] = node

impdf


df2.columns = ['feature_index']
impdf2 = df2.merge(importance_df, on=['feature_index'], how='inner')
impdf2['Node'] = node

impdf2

df3.columns = ['feature_index']
impdf3 = df3.merge(importance_df, on=['feature_index'], how='inner')
impdf3['Node'] = node

impdf3

f1 = pd.read_csv(
    "mnist_fi/feature_split_1/run_0_numhidden_100_lr_1.0E-5_networksize_10_randomseed_12345/vpnn_results_temp_10.csv")


f2 = pd.read_csv(
    "mnist_fi/feature_split_2/run_0_numhidden_100_lr_1.0E-5_networksize_10_randomseed_12345/vpnn_results_temp_10.csv")

f3 = pd.read_csv(
    "mnist_fi/feature_split_1/run_0_numhidden_100_lr_1.0E-5_networksize_10_randomseed_12345/vpnn_results_temp_10.csv")

f1df = f1[f1.Converged == " true"].groupby(['Node']).agg({"TestAUC": "max"}).reset_index()
f1df['Node'] = f1df['Node'].astype(int)

f2df = f2[f2.Converged == " true"].groupby(['Node']).agg({"TestAUC": "max"}).reset_index()
f2df['Node'] = f2df['Node'].astype(int)

f3df = f3[f3.Converged == " true"].groupby(['Node']).agg({"TestAUC": "max"}).reset_index()
f3df['Node'] = f3df['Node'].astype(int)


impdfgrp = impdf.groupby(['Node']).agg({'imp_rank': 'mean'})
impdfgrp2 = impdf2.groupby(['Node']).agg({'imp_rank': 'mean'})
impdfgrp3 = impdf3.groupby(['Node']).agg({'imp_rank': 'mean'})

impdfgrp = impdfgrp.reset_index()
impdfgrp2 = impdfgrp2.reset_index()
impdfgrp3 = impdfgrp3.reset_index()

impdfgrp.merge(f1df, on=['Node'], how='inner')

impdfgrp2.merge(f2df, on=['Node'], how='inner')

impdfgrp3.merge(f3df, on=['Node'], how='inner')