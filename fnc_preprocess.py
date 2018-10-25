import pandas as pd

training_ids = pd.read_csv("./splits/training_ids.txt")
hold_out_ids = pd.read_csv("./splits/hold_out_ids.txt")

train_stances = pd.read_csv("./dataset/train_stances.csv")
train_stances_dict = dict(zip(train_stances['Headline'], train_stances['Body ID'], train_stances['Stance']))

for i in range(training_ids):
    
