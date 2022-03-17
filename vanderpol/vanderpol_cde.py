#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  1 12:01:30 2021

@author: manzand
"""

import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import savemat
from scipy.io import loadmat
import torchcde
import math


parser = argparse.ArgumentParser('ODE demo')
# parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=500)
parser.add_argument('--batch_time', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--niters', type=int, default=10000)
# parser.add_argument('--test_freq', type=int, default=20)
# parser.add_argument('--viz', action='store_true', default=True)
# parser.add_argument('--gpu', type=int, default=0)
# parser.add_argument('--tol', type=float, default=1e-3)
# parser.add_argument('--adjoint', action='store_true', default=True)
args = parser.parse_args()

# if args.adjoint:
#     from torchdiffeq import odeint_adjoint as odeint
# else:
#     from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

np.random.seed(2021)
torch.random.manual_seed(2021)

data = loadmat('/home/manzand/Documents/MATLAB/neuralODE/vanderpol/vdp_data_trace.mat')
tvec = data['tvec'][0]
states = data['states'][0]

# y_test = states[0] # Use the first trajectory for testing
# t_test = tvec[0]

# true_y = torch.tensor(y_test)
# t = torch.tensor(t_test).flatten()
# # t = t[:500] # Reduce training data to first 2500 points
# # true_y = true_y[:500,:]
# true_y0 = torch.tensor(true_y[0])


# def get_batch():
#     s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
#     batch_y0 = true_y[s]  # (M, D)
#     batch_t = t[:args.batch_time]  # (T)
#     batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
#     return batch_y0.to(device), batch_t.to(device), batch_y.to(device)

# y_test = states[0] # Use the first trajectory for testing
# t_test = tvec[0]

# true_y = torch.tensor(y_test)
# t = torch.tensor(t_test).flatten()
# true_y0 = torch.tensor(true_y[0])
t  = []
true_y = []
true_y0 = []
val_exps = 5
for ttt in range(val_exps):
    true_y0.append(torch.tensor(states[ttt][0]).to(device)) 
    t.append(torch.tensor(tvec[ttt][:]).flatten().to(device))
    true_y.append(torch.tensor(states[ttt][:]).to(device))


def get_batch():
    s = np.random.randint(11,len(states),size = (1,10), dtype = int)
    s = list(s[0][:])
    batch_y0 = []
    batch_t = []
    batch_y = []
    for ii in s:
        batch_y0.append(torch.tensor(states[ii][0]).to(device)) 
        batch_t.append(torch.tensor(tvec[ii][:]).flatten().to(device))
        batch_y.append(torch.tensor(states[ii][:]).to(device))
    #     batch_y0.append(states[ii][0]) 
    #     batch_t.append(tvec[ii][:minL])
    #     batch_y.append(states[ii][:minL])
    # batch_y0 = torch.tensor(batch_y0)
    # batch_y = torch.tensor(batch_y)
    # batch_t = torch.tensor(batch_t)
    return batch_y0, batch_t, batch_y

######################
# A CDE model looks like
#
# z_t = z_0 + \int_0^t f_\theta(z_s) dX_s
#
# Where X is your data and f_\theta is a neural network. So the first thing we need to do is define such an f_\theta.
# That's what this CDEFunc class does.
# Here we've built a small single-hidden-layer neural network, whose hidden layer is of width 128.
######################
class CDEFunc(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels):
        ######################
        # input_channels is the number of input channels in the data X. (Determined by the data.)
        # hidden_channels is the number of channels for z_t. (Determined by you!)
        ######################
        super(CDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = torch.nn.Linear(hidden_channels, 128)
        self.linear2 = torch.nn.Linear(128, input_channels * hidden_channels)

    ######################
    # For most purposes the t argument can probably be ignored; unless you want your CDE to behave differently at
    # different times, which would be unusual. But it's there if you need it!
    ######################
    def forward(self, t, z):
        # z has shape (batch, hidden_channels)
        z = self.linear1(z)
        z = z.relu()
        z = self.linear2(z)
        ######################
        # Easy-to-forget gotcha: Best results tend to be obtained by adding a final tanh nonlinearity.
        ######################
        z = z.tanh()
        ######################
        # Ignoring the batch dimension, the shape of the output tensor must be a matrix,
        # because we need it to represent a linear map from R^input_channels to R^hidden_channels.
        ######################
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        return z


######################
# Next, we need to package CDEFunc up into a model that computes the integral.
######################
class NeuralCDE(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels):
        super(NeuralCDE, self).__init__()

        self.func = CDEFunc(input_channels, hidden_channels)
        self.initial = torch.nn.Linear(input_channels, hidden_channels)
        self.readout = torch.nn.Linear(hidden_channels, output_channels)

    def forward(self, coeffs):
        X = torchcde.NaturalCubicSpline(coeffs)

        ######################
        # Easy to forget gotcha: Initial hidden state should be a function of the first observation.
        ######################
        X0 = X.evaluate(X.interval[0])
        z0 = self.initial(X0)

        ######################
        # Actually solve the CDE.
        ######################
        z_T = torchcde.cdeint(X=X,
                              z0=z0,
                              func=self.func,
                              t=X.interval)

        ######################
        # Both the initial value and the terminal value are returned from cdeint; extract just the terminal value,
        # and then apply a linear map.
        ######################
        z_T = z_T[:, 1]
        pred_y = self.readout(z_T)
        return pred_y


######################
# Now we need some data.
# Here we have a simple example which generates some spirals, some going clockwise, some going anticlockwise.
######################
def get_data():
    t = torch.linspace(0., 4 * math.pi, 100)

    start = torch.rand(128) * 2 * math.pi
    x_pos = torch.cos(start.unsqueeze(1) + t.unsqueeze(0)) / (1 + 0.5 * t)
    x_pos[:64] *= -1
    y_pos = torch.sin(start.unsqueeze(1) + t.unsqueeze(0)) / (1 + 0.5 * t)
    x_pos += 0.01 * torch.randn_like(x_pos)
    y_pos += 0.01 * torch.randn_like(y_pos)
    ######################
    # Easy to forget gotcha: time should be included as a channel; Neural CDEs need to be explicitly told the
    # rate at which time passes. Here, we have a regularly sampled dataset, so appending time is pretty simple.
    ######################
    X = torch.stack([t.unsqueeze(0).repeat(128, 1), x_pos, y_pos], dim=2)
    y = torch.zeros(128)
    y[:64] = 1

    perm = torch.randperm(128)
    X = X[perm]
    y = y[perm]

    ######################
    # X is a tensor of observations, of shape (batch=128, sequence=100, channels=3)
    # y is a tensor of labels, of shape (batch=128,), either 0 or 1 corresponding to anticlockwise or clockwise respectively.
    ######################
    return X, y


if __name__ == '__main__':
    num_epochs = 10
    train_X, train_y = get_data()

    ######################
    # input_channels=3 because we have both the horizontal and vertical position of a point in the spiral, and time.
    # hidden_channels=8 is the number of hidden channels for the evolving z_t, which we get to choose.
    # output_channels=1 because we're doing binary classification.
    ######################
    model = NeuralCDE(input_channels=3, hidden_channels=8, output_channels=1)
    optimizer = torch.optim.Adam(model.parameters())

    ######################
    # Now we turn our dataset into a continuous path. We do this here via natural cubic spline interpolation.
    # The resulting `train_coeffs` is a tensor describing the path.
    # For most problems, it's probably easiest to save this tensor and treat it as the dataset.
    ######################
    train_coeffs = torchcde.natural_cubic_coeffs(train_X)

    train_dataset = torch.utils.data.TensorDataset(train_coeffs, train_y)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch_coeffs, batch_y = batch
            pred_y = model(batch_coeffs).squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_y, batch_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print('Epoch: {}   Training loss: {}'.format(epoch, loss.item()))

    test_X, test_y = get_data()
    test_coeffs = torchcde.natural_cubic_coeffs(test_X)
    pred_y = model(test_coeffs).squeeze(-1)
    binary_prediction = (torch.sigmoid(pred_y) > 0.5).to(test_y.dtype)
    prediction_matches = (binary_prediction == test_y).to(test_y.dtype)
    proportion_correct = prediction_matches.sum() / test_y.size(0)
    print('Test Accuracy: {}'.format(proportion_correct))

