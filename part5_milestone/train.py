import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
from model import ResNet
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.nn import *
from datetime import datetime

# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
seed = 42

# valid_split_ratio = 0.2
valid_split_ratio = 0.05

BATCH_SIZE = 64
epochs = 300
# LOSS_FUNCTION = BCEWithLogitsLoss()
LOSS_FUNCTION = BCELoss()
lr = 8e-6

current_time = str(datetime.now())

df = pd.read_csv('data.csv', sep=';')
train_dataset, valid_dataset = train_test_split(df, test_size=valid_split_ratio, random_state=seed)

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
train_data_object = ChallengeDataset(train_dataset, "train")
valid_data_object = ChallengeDataset(valid_dataset, "val")

train_loader = t.utils.data.DataLoader(dataset=train_data_object, batch_size=BATCH_SIZE, drop_last=False, shuffle=True, num_workers=4)
valid_loader = t.utils.data.DataLoader(dataset=valid_data_object, batch_size=BATCH_SIZE, drop_last=True, shuffle=False, num_workers=4)


# create an instance of our ResNet model
MODEL = ResNet()

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
# set up the optimizer (see t.optim)
# create an object of type Trainer and set its early stopping criterion


OPTIMIZER = optim.Adam(MODEL.parameters(), lr=lr)

trainer = Trainer(MODEL, LOSS_FUNCTION, OPTIMIZER, train_loader, valid_loader, cuda=True, early_stopping_patience=20)

# go, go, go... call fit on trainer
res = trainer.fit(epochs)

#save the model
model_name = 'onnx_models/'+current_time + '_final_model.onnx'
trainer.save_onnx(model_name)

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')