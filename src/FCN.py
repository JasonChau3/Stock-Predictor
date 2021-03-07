#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from featureSpaceDay import *


num_days = 5
learning_rate = 0.001
input_size = 4*num_days
hidden_size = 16
num_classes = 1
NUM_EPOCHS = 100

# In[16]:


# Fully connected neural network
class fcn(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(fcn, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = fcn(input_size, hidden_size, num_classes)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#training loop
for e in range(NUM_EPOCHS):
    #use 70% of days for training
    epoch_loss = 0
    for i in range(int(127 * .7)):
        features, labels = featureDaySpace(num_days, i)
        features = torch.FloatTensor(np.array(features))
        model.train()
        optimizer.zero_grad()
        output = model(features)
        loss_train = criterion(output, labels)
        acc_train = accuracy(output, labels)
        loss_train.backward()
        optimizer.step()
        epoch_loss += train_loss.item()
    epoch_loss /= int(127 * .7)
    print(f'Loss for epoch {e}: {epoch_loss}')



def accuracy(preds, labels):

    preds = torch.round(torch.sigmoid(preds))

    acc = torch.round((preds == labels).sum().float() / labels.shape[0] * 100)

    return acc

#test loop
test_losses = []
test_accs = []
for i in range(int(127 * .7) + 1, 127 - num_days -1):
    features, labels = featureDaySpace(i,num_days)
    features = torch.FloatTensor(np.array(features))
    labels = torch.FloatTensor(np.array(labels))
    model.eval()
    output = model(features)
    labels = labels.unsqueeze(1).float()
    loss_test = criterion(output, labels)
    acc_test = accuracy(output, labels)
    test_losses.append(loss_test.item())
    test_accs.append(acc_test.item())
total_test_loss = np.mean(test_losses)
total_test_acc = np.mean(test_accs)

print('This is our total test loss: ' + str(total_test_loss))
print('This is our test accuracy: ' + str(total_test_acc))
