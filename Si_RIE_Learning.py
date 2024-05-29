#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:18:52 2024

@author: shoubhaniknath

    This code is optimized to run on gpu
"""

import torch
import torch.nn as nn
import torch.utils.data
from torchdiffeq import odeint
import numpy as np
import pandas as pd
import sys
sys.path.append("/global/home/users/shoubhaniknath/Python_Codes/")
import ale_learning_RIE as ALE
import time as runtime

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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("cuda.txt",'w') as f:
    f.write(f'Is Cuda available: {torch.cuda.is_available()}\n')
    
# Rate constants represented by neural nets
kc1 = NeuralNet().to(device); kc2 = NeuralNet().to(device); kc3 = NeuralNet().to(device); ks3 = NeuralNet().to(device)

criterion = nn.MSELoss().to(device)

#weight_list = list(kc1.parameters())+list(kc2.parameters())+list(kc3.parameters())+list(ks3.parameters())
weight_list = list(kc1.parameters())+list(kc2.parameters())+list(kc3.parameters())+list(ks3.parameters())

optimizer = torch.optim.Adagrad(params=weight_list)

## Define some system parameters
tsim = torch.tensor(50.0, dtype=torch.float32)                                                 # tsim=0.7 is Non-dimensional time
D = torch.tensor(1e-19, dtype=torch.float32)              # Units of m^2/s: Ho, Surf Sci (1978) 253-263
A = 1063e-20            # Area of surface (from simulation) in m^2
n_Si = torch.tensor(5e28, dtype=torch.float32)             # Density of Si in number/m^3
n_Surface = torch.tensor(72/1063e-20, dtype=torch.float32) # 72 atoms in plane Vella et al, J Vac Sci Technol (2022)
flux_ratio = torch.tensor(5, dtype=torch.float32)          # Value used in our simulation 
v_guess = torch.tensor(1e-11, dtype=torch.float32)

## Space domain discretization parameters
xmin = 0; xmax = 0.1
delX = torch.tensor(1e-2, dtype=torch.float32)
xvec = np.arange(xmin, xmax, delX)
print(f'Number of grid points = {len(xvec)}')

# Package system parameters to pass to the ODE solver
rate_list = [kc1, kc2, kc3, ks3]
rate_list = rate_list

loss_list = []  

ep=0

# Necessary step, declare the number of states
ALE.prob_no = 3
ALE.include_fluence = True; ALE.include_states  = False

energy_list_train = np.arange(20,81,2)
fluence_list_train = np.arange(500,1001,25)
energy_list_str = [str(i) for i in energy_list_train]
fluence_list_str = [str(i) for i in fluence_list_train]

# Initial condition vector for n grid points of atomic fraction in mixed layer
c0 = torch.zeros((len(energy_list_train), len(fluence_list_train), len(xvec)), dtype=torch.float32) 
start = runtime.time()
current_time = start

'---Should have only one loop of epoch---'

directory = '/global/scratch/users/shoubhaniknath/Si_RIE_scratch/Ensemble Average Data/'

# Preprocess data
t, y_sim, init_val, energy, fluence = [tensor.to('cuda') if torch.cuda.is_available() else tensor for tensor in ALE.DataPreSiALE(directory, fluence_list_str, energy_list_str, c0, tsim=tsim, Diffusion=D, v_guess=v_guess)]

params = [tsim.to(device), D.to(device), v_guess.to(device), torch.tensor(A, dtype=torch.float32).to(device), n_Si.to(device), n_Surface.to(device), flux_ratio.to(device), delX.to(device)]
    

for epoch in range(1001):
    
    #torch.autograd.set_detect_anomaly(True)
    args = [fluence, energy, rate_list, params]
    y_pred = odeint(lambda t,y: ALE.SiALE_ODEfunc(t, y, args).to(device), init_val, t, method='dopri5', rtol=1e-5, atol=1e-7).to(device)
    
    
    loss = criterion(y_pred[...,-4:-1], y_sim[...,:-1].to(device))
    if epoch%10==0:
        print(f'Cumulative Epoch = {ep}, Loss = {loss.item():.5f}')
    ep+=1
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with open("loss.txt","a") as f:
        f.write(f'Epoch: {ep},  Loss: {loss.item()}\n')
    
    loss_list.append(loss.item())
        
print(f'Total learning time = {runtime.time()-start}')        

torch.save({"net1_state_dict": kc1.state_dict(),
            "net2_state_dict": kc2.state_dict(),
            "net3_state_dict": kc3.state_dict(),
            "net4_state_dict": ks3.state_dict()}, "/global/home/users/shoubhaniknath/Python_Codes/Model_excluding_calculation_of_Nf.pth") 

loss_df = pd.DataFrame(zip(range(ep), loss_list), columns=['Epoch','Loss'])
loss_df.to_csv("/global/home/users/shoubhaniknath/Python_Codes/Loss_Model_excluding_calculation_of_Nf_RIE.csv")
