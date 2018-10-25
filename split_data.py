import dataset
import generate_test_splits as g
import pandas as pd

d = dataset.DataSet() 
g.generate_hold_out_split(d)

dataset_bodies = pd.read_csv("./dataset/train_bodies.csv")
dataset_stances = pd.read_csv("./dataset/train_stances.csv")
#train_bodies = pd.read_csv("./dataset/competition_test_bodies.csv")
#data_stances = pd.read_csv("./dataset/competition_test_stances_unlabeled.csv")
merged = pd.merge(dataset_bodies, dataset_stances, on="Body ID")
training_bodies = pd.read_csv("./splits/training_ids.csv")
training_bodies_final = merged.loc[merged['Body ID'].isin(training_bodies['Body ID'])]