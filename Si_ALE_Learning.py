#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:18:52 2024

@author: shoubhaniknath

    Code to learn the model parameters of Si ALE. Fluence values are also included in this.
"""

import torch
import torch.nn as nn
import torch.utils.data
import pandas as pd
from torchdiffeq import odeint
import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append("/global/home/users/shoubhaniknath/Python_Codes/")
import ale_learning_2 as ALE
import time as runtime
import random

# Pseudo code:
    # define the neural network to use
    # instantiate the NN and other relevant instances
    # create vector of ion energy values and list of impact values
    # Training loop:
        # Loop over the impact values
        # For each impact value, prepare the concatenated data
        # In each epoch, train the NN model

## Create the learning models
# Custom activation function
class WeightedSigmoid(nn.Module):
    def __init__(self, in_feature):
        super(WeightedSigmoid, self).__init__()
        self.in_feature = in_feature
        self.weight = nn.Parameter(torch.ones(in_feature))
        self.weight.requires_grad = True
                                
    def forward(self, inp):
        activation = self.weight / (1.0 + torch.exp(-inp))
        return activation

# Define the neural network architecture with custom activation
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        
        self.activation = WeightedSigmoid(in_feature=1)
        self.transition_rate = nn.Sequential(
                    nn.Linear(2,128),
                    nn.Tanh(),
                    nn.Linear(128,128),
                    nn.Tanh(),
                    nn.Linear(128,1)
            )
        
    def forward(self, x):
        return self.activation(self.transition_rate(x))

# Using a simple structure without custom activation function will lead to a model with large training loss. I have noticed that there is not good training.

## Instantiate models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("cuda.txt",'w') as f:
    f.write(f'Is Cuda available: {torch.cuda.is_available()}')
    
# Rate constants represented by neural nets
kc1 = NeuralNet().to(device); kc3 = NeuralNet().to(device); ks3 = NeuralNet().to(device)

criterion = nn.MSELoss().to(device)

weight_list = list(kc1.parameters())+list(kc3.parameters())+list(ks3.parameters())

optimizer = torch.optim.Adagrad(params=weight_list, lr=0.01)

## Define some system parameters
tsim = torch.tensor(30.0, dtype=torch.float32)                                                 # tsim=0.7 is Non-dimensional time
D = torch.tensor(1e-19, dtype=torch.float32)              # Units of m^2/s: Ho, Surf Sci (1978) 253-263
A = 1063e-20            # Area of surface (from simulation) in m^2
n_Si = torch.tensor(5e28, dtype=torch.float32)             # Density of Si in number/m^3
n_Surface = torch.tensor(72/1063e-20, dtype=torch.float32) # 72 atoms in plane Vella et al, J Vac Sci Technol (2022)
Nc = torch.tensor(A * 5.8e18, dtype=torch.float32)         # Simulation area = 1063 A^2: Vella et al, J Vac Sci Technol (2022). Active site conc 5.8e18 molecules/m^2: Gao, J Chem Phys (1993)

## Space domain discretization parameters
xmin = 0; xmax = 0.5
delX = torch.tensor(1e-2, dtype=torch.float32)
xvec = np.arange(xmin, xmax, delX)
print(f'Number of grid points = {len(xvec)}')

# Package system parameters to pass to the ODE solver
rate_list = [kc1, kc3, ks3]
rate_list = rate_list

loss_list = []  
v_guess = torch.tensor(1e-11, dtype=torch.float32)
ep=0

# Necessary step, declare the number of states
ALE.prob_no = 3
ALE.include_fluence = True; ALE.include_states  = False

energy_list_train = np.arange(30,71,1)
fluence_list_train = np.arange(100,1001,50)
energy_list_str = [str(i) for i in energy_list_train]
fluence_list_str = [str(i) for i in fluence_list_train]

# Initial condition vector for n grid points of atomic fraction in mixed layer
c0 = torch.zeros((len(energy_list_train), len(fluence_list_train), len(xvec)), dtype=torch.float32) 
start = runtime.time()
current_time = start

'---Should have only one loop of epoch---'
directory = '/global/scratch/users/shoubhaniknath/Ensemble Average Data/'
# Preprocess data
#t, y_sim, init_val, energy, fluence = ALE.DataPreSiALE(directory, fluence_list_str, energy_list_str, c0, tsim=tsim, Diffusion=D, v_guess=v_guess)
t, y_sim, init_val, energy, fluence = [tensor.to('cuda') if torch.cuda.is_available() else tensor for tensor in ALE.DataPreSiALE(directory, fluence_list_str, energy_list_str, c0, tsim=tsim, Diffusion=D, v_guess=v_guess)]
params = [tsim.to(device), D.to(device), torch.tensor(A, dtype=torch.float32).to(device), n_Si.to(device), n_Surface.to(device), Nc.to(device), delX.to(device)]

with open("cuda.txt","a") as f:
    f.write(f'Is cuda available: {torch.cuda.is_available()}')

for epoch in range(2001):
    args = [fluence, energy, rate_list, params]
    y_pred = odeint(lambda t,y: ALE.SiALE_ODEfunc(t, y, args).to(device), init_val, t).to(device)
    
    loss = criterion(y_pred[...,-3:], y_sim.to(device))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    loss_list.append(loss.item())
        
    if epoch%10==0:
        with open("loss_log.txt","a") as f:
            f.write(f'Epoch = {ep}, Loss = {loss.item()}\n')
    ep+=1

print(f'Total learning time = {runtime.time()-start}')       
 

#torch.save({"net1_state_dict": kc1.state_dict(),
#            "net2_state_dict": kc3.state_dict(),
#            "net3_state_dict": ks3.state_dict()}, "/global/home/users/shoubhaniknath/Training_Data/Learned_Model_Including_Fluence_v_1e_11_D_1e_19.pth") 

loss_df = pd.DataFrame(zip(range(ep), loss_list), columns=['Epoch','Loss'])
#loss_df.to_csv("/global/home/users/shoubhaniknath/Training_Data/Loss_with_fluence_v_1e_11_D_1e_19.csv")
