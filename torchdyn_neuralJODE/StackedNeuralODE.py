#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 15:10:59 2021

@author: manzand
"""

'''
Stacked Neural ODEs with Discrete State Transitions
https://torchdyn.readthedocs.io/en/latest/tutorials/01_neural_ode_cookbook.html#Stacked-Neural-ODEs-with-Discrete-State-Transitions
'''
import torch.nn as nn
import torch
from torchdyn.nn.utils import DataControl
from torchdyn.models import NeuralDE
import matplotlib.pyplot as plt
import torch.utils.data as data
import pytorch_lightning as pl
from torchdyn.datasets import ToyDataset

# Set device
device = torch.device('cpu')


# Get data
# Data: we use again the moons dataset (with some added noise) simply because 
# all the models will be effective to solve this easy binary classification problem.
d = ToyDataset()
X, yn = d.generate(n_samples=512, dataset_type='moons', noise=.1)

colors = ['orange', 'blue']
fig = plt.figure(figsize=(3,3))
ax = fig.add_subplot(111)
for i in range(len(X)):
    ax.scatter(X[i,0], X[i,1], s=1, color=colors[yn[i].int()])
    
X_train = torch.Tensor(X).to(device)
y_train = torch.LongTensor(yn.long()).to(device)
train = data.TensorDataset(X_train, y_train)
trainloader = data.DataLoader(train, batch_size=len(X), shuffle=True)

# Define Learner
class Learner(pl.LightningModule):
    def __init__(self, model:nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.005)

    def train_dataloader(self):
        return trainloader

# We choose to divide the domain [0,1] in num_pieces=5 intervals
num_pieces = 5

# stacked depth-invariant Neural ODEs
nde = []
for i in range(num_pieces):
    # nde.append(NeuralDE(nn.Sequential(DataControl(),
    #                                   nn.Linear(4, 4),
    #                                    # nn.Tanh(),
    #                                   nn.Linear(4, 2)), solver='dopri5'))
    nde.append(NeuralDE(nn.Sequential(nn.Linear(2, 4),
                                       # nn.Tanh(),
                                      nn.Linear(4, 2)), solver='dopri5'))
    # In this case the state "jump" is parametrized by a simple linear layer
    nde.append(
        nn.Linear(2, 2)
    )

model = nn.Sequential(*nde).to(device)

learn = Learner(model)
trainer = pl.Trainer(min_epochs=200, max_epochs=250)
trainer.fit(learn)

# Plots

# Evaluate the data trajectories
tf = 20
s_span = torch.linspace(0,1,tf)
trajectory = [model[0].trajectory(X_train, s_span)]
i = 2
c = 0
while c < num_pieces-1:
    x0 = model[i-1](trajectory[c][-1,:,:])
    trajectory.append(
        model[i].trajectory(x0, s_span))
    i += 2
    c += 1

trajectory = torch.cat(trajectory, 0).detach().cpu()
tot_s_span = torch.linspace(0, 5, tf*num_pieces)
# plt.figure()
# plt.plot(tot_s_span,trajectory[:,0,1],'b')

color=['orange', 'blue']

fig = plt.figure(figsize=(10,2))
ax0 = fig.add_subplot(121)
ax1 = fig.add_subplot(122)
for i in range(500):
    ax0.plot(tot_s_span, trajectory[:,i,0], color=color[int(yn[i])], alpha=.1);
    ax1.plot(tot_s_span, trajectory[:,i,1], color=color[int(yn[i])], alpha=.1);
ax0.set_xlabel(r"$s$ [Depth]") ; ax0.set_ylabel(r"$h_0(s)$")
ax1.set_xlabel(r"$s$ [Depth]") ; ax1.set_ylabel(r"$z_1(s)$")
ax0.set_title("Dimension 0") ; ax1.set_title("Dimension 1")

# Trajectories in the depth domain (These functions are never defined)
# plot_2D_depth_trajectory(tot_s_span, trajectory, yn, len(X))

# # Trajectories in the state-space
# plot_2D_state_space(trajectory, yn, len(X))

# # Trajectories in space-depth
# plot_2D_space_depth(tot_s_span, trajectory, yn, len(X))