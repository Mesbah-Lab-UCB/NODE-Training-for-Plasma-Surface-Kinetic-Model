#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 17:09:52 2024

@author: shoubhaniknath

        Python script to validate learned model for Si ALE with Cl/Ar for any condition of Ar ion fluence and energy

        There are some redundancies in this script, e.g., not all the variables are being sent to device, and hence 
        this can be used only on CPU (which is fine for model validation)

        Currently, this script can only be used for one condition at a time. 
"""

# Load the model
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchdiffeq import odeint
import sys
sys.path.append("/Users/shoubhaniknath/my_modules/")
import ale_learning as ALE
import random

## Define the neural network model
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

# Define the neural network architecture
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

# def set_random_seed(seed_value):
#     torch.manual_seed(seed_value)
#     torch.cuda.manual_seed_all(seed_value)  # if using CUDA
#     random.seed(seed_value)
#     np.random.seed(seed_value)

# Set the random seed
#seed = 42  # You can use any integer value as the seed
#set_random_seed(seed)    
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

kc1 = NeuralNet().to(device)

kc3 = NeuralNet().to(device)

ks3 = NeuralNet().to(device)

kc1.load_state_dict(torch.load('Learned_Model_Including_Fluence_v_1e_11_D_1e_19.pth')['net1_state_dict'])
kc3.load_state_dict(torch.load('Learned_Model_Including_Fluence_v_1e_11_D_1e_19.pth')['net2_state_dict'])
ks3.load_state_dict(torch.load('Learned_Model_Including_Fluence_v_1e_11_D_1e_19.pth')['net3_state_dict'])

impacts = ['642']; energies = ['25']
#directory = '/Users/shoubhaniknath/Documents/Rit PhD/NODE/TorchDiffEq Codes/Bombardment Data/Data '\
#    +impacts+' impacts energy '+energies[0]+'/'
directory = '/Users/shoubhaniknath/Documents/Rit PhD/Microscopic Model of Si ALE/Analysis_Energy and Fluence Model/Test Data/Test Data Extracted/'  #energy and fluence
#directory = '/Users/shoubhaniknath/Documents/Rit PhD/NODE/TorchDiffEq Codes/Bombardment Data/3 Sim Ensemble Data/Ensemble Data/'

ALE.prob_no = 3

## Space domain discretization 
xmin = 0; xmax = 0.5
delX = torch.tensor(1e-2, dtype=torch.float32)
xvec = np.arange(xmin, xmax, delX)

#c0 = torch.zeros((len(energies),len(xvec)))  # only energy
# Initial condition vector for n grid points of atomic fraction in mixed layer
c0 = torch.zeros((len(impacts), len(energies), len(xvec)), dtype=torch.float32)  # energy and fluence 

#t, x_test, init_test, energy, fluence = ale_learning.DataPreSiALE(directory, impacts, energies, c0)

# Define some system parameters
tsim = 30.0             # Time of experiment in s. This time is taken from Chen et al, Jpn. J. Appl. Phys., Vol. 44, No. 1A (2005)
D = torch.tensor(1e-19, dtype=torch.float32)              # Units of m^2/s: Ho, Surf Sci (1978) 253-263
A = 1063e-20            # Area of surface (from simulation) in m^2
n_Si = torch.tensor(5e28, dtype=torch.float32)             # Density of Si in number/m^3
n_Surface = torch.tensor(72/1063e-20, dtype=torch.float32) # 72 atoms in plane Vella et al, J Vac Sci Technol (2022)
Nc = torch.tensor(A * 5.8e18, dtype=torch.float32)         # Simulation area = 1063 A^2: Vella et al, J Vac Sci Technol (2022). Active site conc 5.8e18 molecules/m^2: Gao, J Chem Phys (1993)
v_guess = 1e-11

# Package system parameters to pass to the ODE solver
constants = [tsim, D, A, n_Si, n_Surface, Nc, delX]

# List of neural network models
rate_list = [kc1, kc3, ks3] 

time, y_actual, init, energy_tensor, fluence_val = ALE.DataPreSiALE(directory, impacts, energies, c0, Diffusion=D, v_guess=v_guess)
ODEFunc = ALE.SiALE_ODEfunc

#params = [fluence_val, energy_tensor, constants, rate_list] # only energy
params = [fluence_val, energy_tensor, rate_list, constants] # fluence and energy

t, y_pred = ALE.ModelEvaluation(ODEFunc, time, init, params, only_energy=False, method='dopri8')

criterion = nn.MSELoss()
val_loss = criterion(y_actual, y_pred[...,-3:])
print(f'Loss on test data = {val_loss.item():.5e}\n')

ALE.StateProbPlot(t, y_pred[...,-3:], y_actual, energies, impacts, ['II','III','I'])

# Some issues with this method
#ALE.GoodnessOfFit(y_actual.squeeze(1), y_pred[:,:,-3:].squeeze(1))

