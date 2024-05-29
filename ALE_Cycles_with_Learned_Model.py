#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 15:48:34 2024

@author: shoubhaniknath

    Python script to simulate cycle dynamics of Si ALE with Cl/Ar using learned model for the transition probabilities.
    The script outputs plots for the Cl uptake and distance of Si etched at different conditions of incident ion fluence 
    and ion energy. 
    
    For accurate results, one must use different guess veloities for different energy conditions. 
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/shoubhaniknath/my_modules/')
import ale_learning as ALE
from torchdiffeq import odeint

## Define some system parameters
tsim = 30.0                                                 # tsim=0.7 is Non-dimensional time
D = torch.tensor(1e-19, dtype=torch.float32)                # Units of m^2/s: Ho, Surf Sci (1978) 253-263
A = 1063e-20                                                # Area of surface (from simulation) in m^2
n_Si = torch.tensor(5e28, dtype=torch.float32)              # Density of Si in number/m^3
n_Surface = torch.tensor(72/1063e-20, dtype=torch.float32)  # 72 atoms in plane Vella et al, J Vac Sci Technol (2022)

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

def velocity_func(t, l, args):
    ion_fluence, ion_energy, t_exp, Nc, n_Si, A, kc1_val, kc3_val, ks3_val, y = args
    ion_fluence = torch.tensor(ion_fluence, dtype=torch.float32).unsqueeze(0)
    ion_energy = torch.tensor(ion_energy, dtype=torch.float32).unsqueeze(0)
    flux = ion_fluence / (t_exp * A)
    nn_input = torch.stack((ion_fluence, ion_energy), dim=1)
    v = (flux * Nc / n_Si) * ((kc1_val(nn_input) + kc3_val(nn_input) - \
                               ks3_val(nn_input)) * y[...,0] + \
                              ks3_val(nn_input) * A * n_Surface / Nc)
    return v.squeeze(0)

def is_increasing(tensor):
    # Check if all elements are in increasing order
    return torch.all(tensor[:-1] <= tensor[1:])

def euler(func_val, step_size, y0, n):
    y = torch.zeros((n))
    y[0] = y0
    for i in range(n-1):
        y[i+1] = y[i] + step_size * func_val[i]
    return y

# Load the model and disable gradient computations
torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

kc1 = NeuralNet()

kc3 = NeuralNet()

ks3 = NeuralNet()

kc1.load_state_dict(torch.load('Learned_Model_Including_Fluence_v_1e_11_D_1e_19.pth')['net1_state_dict'])
kc3.load_state_dict(torch.load('Learned_Model_Including_Fluence_v_1e_11_D_1e_19.pth')['net2_state_dict'])
ks3.load_state_dict(torch.load('Learned_Model_Including_Fluence_v_1e_11_D_1e_19.pth')['net3_state_dict'])

impacts = [1000.]; energies = [float(i) for i in np.arange(40,101,20)]

ALE.prob_no = 3

## Space domain discretization 
xmin = 0; xmax = 0.5
delX = torch.tensor(1e-2, dtype=torch.float32)
xvec = np.arange(xmin, xmax, delX)

# List of neural network models
rate_list = [kc1, kc3, ks3] 

#time, y_actual, init, energy_tensor, fluence_val = ALE.DataPreSiALE(directory, impacts, energies, c0)
ODEFunc = ALE.SiALE_ODEfunc

# DEfine the total ALE cycles and the timesteps for ODE solver
cycles = 8; steps = 100

# List to store the total number of Cl atoms if required
Nc_list=[]

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

# List to store the Cl occupation numbers for each ion energy value
c_final = []

for count, energy in enumerate(energies): 
    
    ## Initialisation
    # Initial condition vector for n grid points of atomic fraction in mixed layer
    c0 = torch.zeros((len(impacts), 1, len(xvec)), dtype=torch.float32) 
    p0 = torch.zeros((len(impacts), 1, ALE.prob_no), dtype=torch.float32)
    p0[...,0] = 1
    
    Nc = torch.tensor(A * 5.8e18, dtype=torch.float32)         # Simulation area = 1063 A^2: Vella et al, J Vac Sci Technol (2022). Active site conc 5.8e18 molecules/m^2: Gao, J Chem Phys (1993)
    n0 = Nc
    v_guess = [0.2e-11, 0.7e-11, 1.2e-11, 1.8e-11]
    time = torch.tensor(np.linspace(0.0, tsim / (D/v_guess[count]**2), steps))
    c_pred = torch.empty((len(time),3))
    cumulative_time = torch.empty((len(time)))
    cumulative_etch = torch.empty((len(time)))
    l0 = torch.tensor([0.], dtype=torch.float32)

    for cycle in range(cycles):
        constants = [tsim, D, A, n_Si, n_Surface, Nc, delX]
        params = [torch.tensor(impacts, dtype=torch.float32), torch.tensor([energy], dtype=torch.float32),\
                  rate_list, constants] # fluence and energy
    
        init = torch.cat((c0,p0), dim=2)
        t, y_pred = ALE.ModelEvaluation(ODEFunc, time, init, params, method='dopri8')
        
        velocity = velocity_func(t, l0, [impacts[0], energy, tsim, Nc, n_Si, \
                                                                A, kc1, kc3, ks3, y_pred[...,-3]])
        etch_pred = euler(velocity, 0.3, l0, steps)
        
        y_prob = (y_pred[...,-3:].squeeze(1)).squeeze(1)
        c_pred = torch.cat((c_pred, Nc * y_prob), dim=0)
        
        # Total time
        cumulative_time = torch.cat((cumulative_time, (cumulative_time[-1]+t)), dim=0)
        
        # Total etch
        cumulative_etch = torch.cat((cumulative_etch, (cumulative_etch[-1]+etch_pred)), dim=0)
        
        # Update the probabilities
        p0 = y_pred[...,-3:][-1]
        
        ## Start of the adsorption cycle
        # Update the total number of Cl atoms. Surface is already filled with Cl atoms
        Nc = n0 + Nc * (p0[...,-1].squeeze())
        
        # Update the mixed layer concentration
        c0 = y_pred[...,:-3][-1]
        
        Nc_list.append(Nc)
        
        # Start of Cl cycle, surface is filled with Cl. Recalculate the probabilities
        p0[...,0] = n0 / (Nc)
        p0[...,1] = 0.              # Bulk gas evacuated before Cl cycle. 
        p0[...,2] = 1-p0[...,0]

    t_final = cumulative_time[100:] * (D/v_guess[count]**2)
    c_final = c_pred[100:,:]
    
    ax1.plot(t_final, (c_final[:,0]+c_final[:,2])/n0)
    cumulative_etch = cumulative_etch.detach().cpu().numpy()
    ax2.plot(t_final, cumulative_etch[100:]/1e-10)
    
tick_pos = np.array([cycle*tsim for cycle in range(cycles+1)])
tick_label = [str(cycle) for cycle in range(cycles+1)]

ax1.set(xlabel = 'Cycles', ylabel = 'Cl Uptake (ML)')
ax1.set_xticks(tick_pos, tick_label)
ax1.legend([str(int(i))+ 'eV' for i in energies], loc='best')
ax1.grid()
fig1.show()

ax2.set(xlabel = 'Cycles', ylabel = 'Si Etched ($\AA$)')
ax2.set_xticks(tick_pos, tick_label)
ax2.legend([str(int(i))+ 'eV' for i in energies], loc='best')
ax2.grid()
fig2.show() 
