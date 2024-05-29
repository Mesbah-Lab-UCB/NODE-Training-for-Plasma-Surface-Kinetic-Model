#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 17:09:52 2024

@author: shoubhaniknath

    Validation of learned model for Si RIE. This code follows a similar one written for ALE.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sys
sys.path.append("/Users/shoubhaniknath/my_modules/")
import ale_learning_RIE as ALE
import matplotlib.pyplot as plt


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

    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

kf1 = NeuralNet(); kf2 = NeuralNet(); kf3 = NeuralNet(); ks3 = NeuralNet()

kf1.load_state_dict(torch.load('Model_excluding_calculation_of_Nf.pth', map_location=torch.device('cpu'))['net1_state_dict'])
kf2.load_state_dict(torch.load('Model_excluding_calculation_of_Nf.pth', map_location=torch.device('cpu'))['net2_state_dict'])
kf3.load_state_dict(torch.load('Model_excluding_calculation_of_Nf.pth', map_location=torch.device('cpu'))['net3_state_dict'])
ks3.load_state_dict(torch.load('Model_excluding_calculation_of_Nf.pth', map_location=torch.device('cpu'))['net4_state_dict'])

impacts = ['750']; energies = ['50']
#directory = '/Users/shoubhaniknath/Documents/Rit PhD/NODE/TorchDiffEq Codes/Bombardment Data/Data '\
#    +impacts+' impacts energy '+energies[0]+'/'
directory = '/Users/shoubhaniknath/Documents/Rit PhD/Surface Kinetic Model/Si RIE with F-Ar/Data Folder/Training Data/'  #energy and fluence

ALE.prob_no = 3

## Space domain discretization 
xmin = 0; xmax = 0.1
delX = torch.tensor(1e-2, dtype=torch.float32)
xvec = np.arange(xmin, xmax, delX)

#c0 = torch.zeros((len(energies),len(xvec)))  # only energy
# Initial condition vector for n grid points of atomic fraction in mixed layer
c0 = torch.zeros((len(impacts), len(energies), len(xvec)), dtype=torch.float32)  # energy and fluence 

#t, x_test, init_test, energy, fluence = ale_learning.DataPreSiALE(directory, impacts, energies, c0)


## Define some system parameters
tsim = torch.tensor(50.0, dtype=torch.float32)             # tsim=0.7 is Non-dimensional time
D = torch.tensor(1e-19, dtype=torch.float32)               # Units of m^2/s: Ho, Surf Sci (1978) 253-263
A = 1063e-20                                               # Area of surface (from simulation) in m^2
n_Si = torch.tensor(5e28, dtype=torch.float32)             # Density of Si in number/m^3
n_Surface = torch.tensor(72/1063e-20, dtype=torch.float32) # 72 atoms in plane Vella et al, J Vac Sci Technol (2022)
flux_ratio = torch.tensor(5, dtype=torch.float32)          # Value used in our simulation 
v_guess = torch.tensor(1e-11, dtype=torch.float32)

# Package system parameters to pass to the ODE solver
constants = [tsim, D, v_guess, A, n_Si, n_Surface, flux_ratio, delX]

# List of neural network models
rate_list = [kf1, kf2, kf3, ks3] 

time, y_actual, init, energy_tensor, fluence_val = ALE.DataPreSiALE(directory, impacts, energies, c0, tsim=tsim, Diffusion=D, v_guess=1e-11)
ODEFunc = ALE.SiALE_ODEfunc

#params = [fluence_val, energy_tensor, constants, rate_list] # only energy
params = [fluence_val, energy_tensor, rate_list, constants] # fluence and energy

t, y_pred = ALE.ModelEvaluation(ODEFunc, time, init, params, only_energy=False, method='bosh3', rtol=1e-3, atol=1e-5)

criterion = nn.MSELoss()
val_loss = criterion(y_pred[...,-4:-1], y_actual[...,:-1])
print(f'Loss on test data = {val_loss.item():.5e}\n')

ALE.StateProbPlot(t, y_pred[...,-4:-1], y_actual[...,-4:-1], energies, impacts[0], ['II','III','I'], D, v_guess)

# ALE.eval_mode = True
# velocity = ALE.SiALE_ODEfunc(t, y_pred, [torch.tensor(float(impacts[0]), dtype=torch.float32), \
#                                          torch.tensor(float(energies[0]), dtype=torch.float32), [kf1, kf2, kf3], \
#                                              [tsim, D, A, n_Si, n_Surface, flux_ratio, delX]])

Nf, dNfdt = ALE.Total_F(t, [D, v_guess, int(impacts[0])/(tsim * A), A])
index_nf = 50 * np.array(range((round(len(t)/50)))); t_nf = t[index_nf] * (D/v_guess**2)

plt.figure()
plt.plot((t * (D/v_guess**2)).detach().numpy() , Nf.detach().numpy(), c='k', label='From Computation')
plt.scatter(t_nf, y_actual[index_nf,:,:,-1], c='k', facecolors='none', label='From Simulation')
plt.xlabel('Time (s)'); plt.ylabel('Number of F atoms'); plt.legend()

velocity = ALE.velocity_func(t, y_pred, params)
#velocity = velocity.view(int(impacts[0]))
time_scale = D / velocity**2
Nf_actual = y_actual[:,0,0,-1]; Nf_pred = y_pred[:,0,0,-1] * time_scale

Nf_1 = torch.mul(Nf, y_pred[...,-2].view((int(impacts[0]))))
Nf_2 = torch.mul(Nf, y_pred[...,-4].view((int(impacts[0]))))
Nf_3 = torch.mul(Nf, y_pred[...,-3].view((int(impacts[0]))))

fig, ax = plt.subplots()
ax.plot((t * (D/v_guess**2)).detach().numpy(), Nf_1.detach().numpy(), label='Mixed Layer', color='r')
ax.plot((t * (D/v_guess**2)).detach().numpy(), Nf_2.detach().numpy(), label='Surface', color='b')
ax.plot((t * (D/v_guess**2)).detach().numpy(), Nf_3.detach().numpy(), label='Bulk Gas', color='g')
ax.set(xlabel='Time (s)', ylabel='Number of F atoms')
ax.legend()

#ALE.GoodnessOfFit(y_actual.squeeze(1), y_pred[:,:,-3:].squeeze(1))

