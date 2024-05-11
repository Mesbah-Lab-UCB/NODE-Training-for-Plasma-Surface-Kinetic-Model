#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 13:46:54 2024

@author: shoubhaniknath
    
    This is a module related to plasma processing (ALE & RIE) for preprocessing data for neural network training, ODE function for space discretisation of a 
    drift-diffusion equation to pass to a ODE solver (used to train neural network in a NODE-framework), and postprocessing of trained model.

    The methods in this module are tailored for the surface-kinetic-drift-diffusion model that has been developed in the Mesbah Lab. This model is a system 
    of coupled ODEs and one PDE (diffusion in solid substrate), and the ODEs are a system of master equations. The transition probabilities of the master 
    equation are represented as neural networks, which need to be trained. The training procedure consists of solving the system of ODEs, and computing the 
    loss from the solution and the MD data. 

    This code has been created with the aim to provide the user with the option to choose whether to include the ion fluence and the state values themselves 
    in the feature vector for the neural network. States are basically the dependent variables of the ODE, and are included when the master equation itself is 
    non-linear in nature. Ion energy is a required feature, as the transition probability must depend on the value of the incident ion energy. However, as 
    the code currently stands, we cannot include the states as features to the neural network, and more work needs to be done in order to include the states
    as features as well. For Si ALE, including the states as features is not required. 

    The data from MD is in tabular form with different .csv files for different conditions, and each .csv file contains time series data of the occupation 
    number at different states. This data needs to be properly preprocessed in order to feed it to the neural network. If the fluence is included, then the 
    data is a 3D tensor, but more importantly, preprocessing needs to be done, since the size of the time series will change with the fluence value. More 
    details are provided in the method. An index of the methods is given below:

    Data Preprocessing: ConcatSimData, DataPreSiALE, MapTime
    ODE function: SiALE_ODEFunc
    Postprocessing: ModelEvaluation, StateProbPlot, GoodnessOfFit
    Miscellaneous: others
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Number of states which have occupation probability
# Must define before calling any method as module_name.prob_no = {}; module_name.method(args)
prob_no = None
include_fluence = True
include_states = False

  
def ConcatSimData(df, tsim, x_data, D, v):
    
    """
        Generates a time series tensor from the dataframe and creates training data for state 
        probabilities. Concatenates feature columns from a dataframe to the tensor. Generates 
        non-dimensional time series data
        
        PARAMETERS: 
        -----------
            df: pandas dataframe
                Dataframe containing MD simulation data of timesteps and state probability values
            tsim: float
                The time of experiment corresponding to total simulation time
            x_data: pytorch tensor
                Tensor to which simulation data must be concatenated to, typically containing 
                data at a different value of ion impact
            D: float
                Diffusion coefficient 
            v: float
                Guess velocity value to get a time scale
        
        RETURNS:
        --------
            time_data: 1D tensor
                Non-dimensionalised and rescaled time series
            x_data: Tensor
                Concatenated tensor
        
    """
    # Scale time data between 0 and 30, and non-dimensionalise. D and v are global variables
    # Shape of 1D tensor is [[ no_of_impacts+1 ]]
    time_scale = D / v**2

    # The first column has to be time value. Rename the first column to time
    cols = df.columns
    df.rename({cols[0]:'Time'}, inplace=True)
    
    # Time series tensor. This is non-dimensional and rescaled time
    time_data = (torch.tensor(np.linspace(0,tsim,len(df['Time'])), dtype=torch.float32)) \
                / time_scale
    
    # 2D Dummy tensor to store data
    # Dim 0: no of impacts+1    Dim 1: Probability variables
    new_data = torch.tensor(df[df.columns[1:prob_no+1]].values, dtype=torch.float32)
    
    # Concatenate the data. The new dataframe matrix goes last into the first dimension
    # The size of the tensor is len(energy values) x No of impacts x len(probabilities)
    x_data = torch.cat([x_data, new_data.unsqueeze(0)], dim=0)
    
    return time_data, x_data
    

def is_nan(tensor):
    # Check if any element in the tensor is NaN
    return torch.isnan(tensor).any()
        

def find_nan_indices(tensor):
    # Create a Boolean mask indicating NaN values
    nan_mask = torch.isnan(tensor)
    
    # Use torch.nonzero() to find the indices of NaN values
    nan_indices = torch.nonzero(nan_mask, as_tuple=False)
    
    return nan_indices


def MapTime(fmin, f_current, tsim, tstart=0):
    
    """
        Maps the time series of any fluence value to that of the minimum fluence 
        value. This is necessary as the ODE solver will take in 1D time tensor. 
        Thus the number of time points for two fluence values will be different. 
        This method gives the indices of any time tensor of any fluence values 
        which are closes to the time corresponding to the minimum fluence value. 
        It essentially maps the time series of any fluence value to that of the 
        smallest fluence value
        
        PARAMETERS:
        ----------
            fmin: int
                Minimum fluence value
            f_current: int
                Fluence value to be mapped
            tsim: float
                The time of experiment corresponding to total simulation time
            tstart: float, optional
                Start time of experiment. Default value is 0
        
        RETURNS:
        -------
            closes_indices: numpy ndarray
                mapping of indices closest to the minimum tensor
    """
    tmin = np.linspace(tstart,tsim,int(fmin)+1)
    t = np.linspace(tstart,tsim,f_current+1)
    
    tmin_reshaped = tmin[:, np.newaxis]
    t_reshaped = t[np.newaxis, :]
    
    abs_diff = np.abs(tmin_reshaped - t_reshaped)
    
    closest_indices = np.argmin(abs_diff, axis=1)
    
    return closest_indices    

    
def DataPreSiALE(path_to_dir, list_of_impacts, list_of_energies, y0_internal, skiprows=False, last_row = 0, tsim=30.0, Diffusion=1e-19, v_guess=1e-11):
    
    """
        Preprocesses the training data. Concatenates the time-series at different ion energies 
        for a single value of ion impact. Generates tensors to be used for model training and 
        evaluation. The name of the .csv file containing the data should be in the format: 
        # impacts # energy_occupation number.csv
        
        The .csv datafile must have the columns as ['Time','Surface','Bulk','Total','Surface Prob','Bulk Prob']
        
        PARAMETERS:
        ----------
            path_to_dir: string
                        Path to the directory which contains the data files
            list_of_impacts: string
                        Number of ion impacts
            list_of_energies: list of string
                        Values of ion energies
            y0_internal: 2D pytorch tensor
                        Tensor of the probability values at all internal grid points. The
                        first dimension (dim 0) is for ion energies and the second dimension 
                        (dim 1) is for the internal values. Length of the tensor in dim 1 
                        is equal to the number of grid points in discretization
            skiprows: Boolean, optional
                    Skip any rows or not, if the data contains values from adsorption as well
            last_row: int, optional
                    Last row till which we have to skip. Assumes that adsorption values are 
                    before bombardment values
            tsim: float, optional
                    The time of experiment corresponding to total simulation time
            Diffusion: float, optional
                        Diffusion coefficient. For dimensional case, use D=1
            v_guess: float, optional
                Guess velocity value to get a time scale. For dimensional case, use v_guess=1
        
        RETURNS:
        -------
            time_data: 1D pytorch tensor
                    Tensor of non-dimensional and rescaled time series
            x_data: 3D pytorch tensor
                    Simulation and internal data at different energies. All the time 
                    series data at different ion energies are concatenated. Dim 0 is 
                    for ion energies, Dim 1 is for time points or number of impacts, 
                    and Dim 3 is for the internal values and state probabilities
            init_data_all: 2D pytorch tensor
                    2D tensor of the initial conditions that have to be passed to the 
                    ODE solver. This is the first value of Dim 1 of x_data. Dim 0 
                    corresponds to the ion energies and Dim 1 corresponds to the 
                    internal values and state probabilities
            energy_tensor: 1D pytorch tensor
                    1D tensor of the ion energy values. Shape is [[no of ion energies]]
            fluence_tensor: 1D pytorch tensor
                    Tensor of the fluence value used. Contains only one element                
    """
    
    # The column rows have to be ['Time']
    # Prepare the initial_condition data, time series data, fluence value, energy tensor
    
    # Tensor of initial internal values in the mixed layer
    # Dim 0: energies   Dim 1: fluence  Dim 2: no of grid points
    y0_internal = y0_internal.clone().detach().to(torch.float32)
    
    if len(y0_internal.shape)!=3:
        raise Exception(f'Expected 3 dimensions, one for energy, one for fluence, and one for dependent values; got {len(y0_internal)}')
    
    # Number of internal grid points
    n_grid = len(y0_internal[0,0,:])
    
    # Tensor to store the time series data of all the probabilities for different energies
    # Dim 0: energies   Dim 1: No of impacts     Dim 2: Probability variables
    list_of_impacts_num = [float(i) for i in list_of_impacts]
    
    # Dim 0: energies   Dim 1: fluence   Dim 2: Time series   Dim 3: Probability variables
    x = torch.empty((0, 0, int(min(list_of_impacts_num)+1), prob_no))
    
    # Dim 0: energies   Dim 1: fluence   Dim 1: probability and internal points
    for f in list_of_impacts:
        x_data = torch.empty((0, int(f)+1, prob_no))
        
        # Loop to create a tensor of the simulation data at different energy values
        for e in list_of_energies:
            
            filename = f+' impacts '+e+' energy_occupation number.csv'
            
            if skiprows:
                df = pd.read_csv(path_to_dir+filename, skiprows=range(1,last_row))
            else:
                df = pd.read_csv(path_to_dir+filename)
         
            df.drop(['Surface', 'Bulk', 'Total'], axis=1, inplace=True)
            df.rename(columns={"Bulk Prob":"Mixed Layer Prob"}, inplace=True)
            df['Bulk Prob'] = 1 - (df['Surface Prob']+df['Mixed Layer Prob'])
            
            # Reposition column
            ml_prob = df.pop('Mixed Layer Prob')
            df.insert(len(df.columns), 'Mixed Layer Prob', ml_prob)
            
            time_data, x_data = ConcatSimData(df, tsim, x_data, Diffusion, v_guess)
            
        if f==list_of_impacts[0]:
            t = time_data
            x = x_data.unsqueeze(1)
            
        else:
            indices = MapTime(list_of_impacts_num[0], int(f), tsim)
            x_new = x_data[:,indices,:]
            x = torch.cat((x, x_new.unsqueeze(1)), dim=1)
            
            
    ## Initial conditions for the probabilities
    # Dim 0: energies   Dim 1: fluence  Dim 2: probability values
    init_probability = x[...,0,:]
    
    # The initial condition tensor should be a 3D tensor of the initial values of the 
    # state probabilities and the internal values
    # 3D tensor of initial values. Concatenate internal values and state probabilities
    # Dim 0: energies   Dim 1: probabilities + grid points
    init_data_all = torch.cat((y0_internal, init_probability), dim=2)
    
    ## 1D Tensor of energy values and fluence value
    energy_tensor = torch.tensor([float(e) for e in list_of_energies], dtype=torch.float32)
    fluence_tensor = torch.tensor([float(f) for f in list_of_impacts], dtype=torch.float32)
    
    # Transpose the data to keep parity with output from ODE solver
    # Not a necessity, just a good practice
    x = torch.transpose(torch.transpose(x, 1, 2), 0, 1)
    
    # time_data: [no of impacts]; x_data: [no of impacts x energies x probability values]
    # init_data_all: [energies x (state probability values + internal values)]
    # energy_tensor: [energies]
    
    return t, x, init_data_all, energy_tensor, fluence_tensor
        

def SiALE_ODEfunc(t, y, args):
    
    """
        Function to discretize the drift-diffusion PDE in Si ALE and pass the time 
        derivative values of internal points and state probabilities. This function 
        is passed to an ODE solver like odeint to solve a system of coupled ODEs. 
        The solved variables are the three state probabilities, and the atomic fraction 
        at the internal grid points. This method is designed to take in a vector of 
        different ion energy values.
        
        Central difference method has been used to discretize the set of equations. 
        Check the overleaf document for the equations used in this method.
        
        PARAMETERS:
        ----------
        t: 1D pytorch tensor of time series values for which to solve for. Must use 
        non-dimensional time
        y: 2D pytorch tensor of variable values. Dim 0 corresponds to the different
            values of the ion energy, and Dim 1 corresponds to the different variables. 
            For n internal grid points, shape of Dim 1 is n+3
        args: list of parameters required in the functions
        
        RETURNS:
        -------
        func: 2D pytorch tensor of the time derivative values
              Dim 0 is for different energy values, Dim 1 is for the internal points 
              and state probabilities. The time derivative values are used by the ODE 
              solver to solve the equation y' = f(t,y) 
    """
    # y here is a len(energy_vec) x len(fluence_vec) x (N+3) tensor
    # velocity will be len(energy_vec) long in this case
    # Unpack the relevant variables
    impact_vec, energy_vec, rate_constants, params = args
    
    # n is the size of the dependent variables. l is for energy vector, and m is for fluence
    l, m, n = y.size()
    
    # Number of internal grid points
    N = n - 3
    
    # Unpack parameters and constants
    tsim, D, A, n_Si, n_Surface, Nc, delX = params
    
    # Unpack NN models for the rate constants
    kc1, kc3, ks3 = rate_constants
    
    # From simulation we only have the fluence, but in the equations, we have flux
    fluence = impact_vec / A
    flux = fluence / tsim
    
    combined_fluence_energy = torch.stack((impact_vec.repeat(len(energy_vec)), energy_vec.repeat(len(impact_vec))), dim=1)
    
    if not include_fluence and not include_states:
        # If only energy is the input. Change size to len(energy)x1 for NN
        model_input = energy_vec.unsqueeze(1)    
    elif include_fluence and not include_states:
        # Both ion energy and ion impact are inputs
        # The input is all combinations of energy and fluence. Shape is len(energy)*len(fluence) x 2
        model_input = combined_fluence_energy
    elif not include_fluence and include_states:
        # Energy and states are inputs. Shape is len(energy) x 3. Dim 0: Energies  Dim 1: Inputs
        model_input = torch.cat((energy_vec.unsqueeze(1), y[...,-3], y[...,-1]), dim=1)
        #model_input = torch.cat((energy_vec, y[:,-3].unsqueeze(1),y[:,-1].unsqueeze(1)), dim=0)
    else: 
        # All three are inputs. Model inputs is a 2D tensor of shape len(energy)*len(fluence) x (len(input states)+1)
        # Basically, we take all combinations of fluence and energy, and the states corresponding to each such combination
        
        # Reshape the states to size len(energy)*len(fluence) x 1
        y1 = y[...,-3].view(l*m,1); y2 = y[...,-3].view(l*m,1)
        
        # concatenate the model inputs
        model_input = torch.cat((combined_fluence_energy, y1, y2), dim=1)
    
    # Find the first linear layer
    first_linear_layer = None
    for layer in kc1.modules():
        if isinstance(layer, nn.Linear):
            first_linear_layer = layer
            break
    
    # Get the number of input features of the first linear layer
    if first_linear_layer:
        input_features = first_linear_layer.in_features
        
    if input_features != model_input.shape[1]:
        raise Exception('Number of features in input is not equal to the number of inputs to the NN. Possible actions \
                        are 1) Check the first layer of the neural network, and 2) Check how many model inputs are needed')
    
    kc1_out = kc1(model_input)
    kc1_val = kc1_out.reshape(l,m)
    kc3_out = kc3(model_input)
    kc3_val = kc3_out.reshape(l,m)
    ks3_out = ks3(model_input)
    ks3_val = ks3_out.reshape(l,m)
   
    v = (flux * Nc / n_Si) * ((kc1_val + kc3_val - \
                               ks3_val) * y[...,N] + \
                              ks3_val * A * n_Surface / Nc)

    if is_nan(v):
        print(kc1)
        print(kc3)
        print(ks3)                    
        raise Exception('Velocity is NaN')
    
    func = torch.zeros(y.size())
    
    # Time derivatice values for the internal points. First index is no of variables, second is no of energies
    func[...,1:N-2] = (y[...,2:N-1] - 2 * y[...,1:N-2] + y[...,0:N-3]) / (delX**2) - (y[...,2:N-1] - y[...,0:N-3]) / (2 * delX)
    
    # Used to calculate fictitious point at all energy values
    alpha = (torch.mul(kc1_val, y[...,N]) * flux * Nc) / (v * n_Si) - y[...,0]
    
    # Atomic fraction value at fictitious point. This is used for central difference of the slope dc/dX at X=0
    # The value of the fictitious point is determined in terms of the boundary and internal points using the 
    # boundary condition at the left boundary
    y_at_minus1 = y[...,1] + 2 * delX * alpha    
    
    # Governing equation and the fictitious point value used to determine the atomic fraction at the boundary
    func[...,0] = (y[...,1] - 2 * y[...,0] + y_at_minus1) / (delX**2) - (y[...,1] - y_at_minus1) / (2 * delX)         
    
    # Surface ODE
    func[...,N] =  ((-D / v**2) * flux * A * y[...,N] * (kc1_val + kc3_val) +\
                (D / v) * (n_Si * A * y[...,0] / Nc))
                
                
    # Bulk Gas ODE, decoupled from the rest of the equations. Used for plotting and sanity checks.
    func[...,N+1] = (flux * A * D * kc3_val * y[...,N] / v**2) # or func[-2]

    # Mixed Layer ODE. This is also decoupled from the rest of the equations. Used for plotting and sanity checks.
    #func[:,N+2] = - ((n_Si * A / Nc) * (D / v) * ((y[:,1].unsqueeze(1) - y_at_minus1) / (2 * delX))).squeeze(1)
    func[...,N+2] = - ((n_Si * A / Nc) * (D / v) * ((y[...,1] - y_at_minus1) / (2*delX)))
    
    if is_nan(func):
        indices = find_nan_indices(func)
        print('Following indices have NaN')
        print(indices)
        print(f'Time is {t}')
        print('The rate constant values are:')
        print(kc1)
        print(kc3)
        print(ks3)
        print(func)
        raise Exception('Derivative has NaN')
        
    return func


def ModelEvaluation(odefunc, time_series, init, params, *args, method='dopri8', rtol=1e-7, atol=1e-8, **kwargs):
    
    """
        Method to evaluate trained model. 
        Calls the DataPreSiALE method to preprocess data. The processed data is then used 
        to generate predictions through a trained model. Uses odeint as an ODE solver
        
        PARAMETERS:
        ----------
            odefunc: function to be passed to ODE solver
                    ODEFunc is the main ODE function that represents the time derivatives 
                    of the state probabilities and the discretized internal points.
            time_series: 1D pytorch tensor
                    Tensor of time series values at which ODE solver needs to solve
            init_data: 2D pytorch tensor
                    Tensor of initial conditions. Dim 0 is for different energy values 
                    and Dim 1 is for the variables to be solved for
            params: list
                    Parameters which need to be passed to the odefunc. This is specific 
                    to the theoretical model and can be constants or system specific 
                    parameters. Relevant models for the learned rate constants can be 
                    passed in this.
            method: string, optional
                    Method to solve ODE. Default is dopri8
            rtol:   float, optional
                    Relative tolerance. Default is 1e-7
            atol:   float, optional
                    Absolute tolerance, should be lower than relative tolerance. Default is 1e-8
            *args:  list
                    Non-keyword arguments for DataPreSiALE() which is called in this method
            **kwargs: dict
                    Dictionary of keyword arguments passed to DataPreSiALE() method. Might 
                    be irrelevant in this usage
                    
        RETURNS:
        -------
            time_data: 1D pytorch tensor
                    Tensor of non-dimensional and rescaled time series
            y_pred: 3D pytorch tensor
                    Tensor of solution values of the system of ODEs. Solution values in dim 0 
                    Shape is [timesteps(no of impacts), energy values, variables]
            y_actual: 3D pytorch tensor
                    Simulation data with shape same as y_pred
                
    """
    
    ' For the Si ALE Etch with Cl/Ar, use the DataPreSiALE method to generate the '
    ' initial value and time series that has to be passed to the ODE solver. *args '
    ' and **kwargs kept here in case they become relevant for future use '
    
    
    # Model evaluation must not have recomputation of gradients
    with torch.no_grad():
        
        # Solve the system of ODEs
        # y_pred is a 3D tensor and solved output at each timestep is appended to the first dimension
        y_pred = odeint(lambda t, x: odefunc(t,x,params), init, time_series, method=method, rtol=rtol, atol=atol)
   
    # The output is a 3D tensor
    # Dim 0: no of impacts     Dim 1: energies     Dim 2: internal and probability values
    return time_series, y_pred


def GoodnessOfFit(y_actual, y_pred):
    
    # Both the actual and predicted data must be no more than 2D
    from torcheval.metrics import R2Score
    metric = R2Score(multioutput='raw_values')
    metric.update(y_actual, y_pred)
    print(metric.compute())
    
    return metric.compute


def StateProbPlot(t, y_pred, y_actual, energy_list, no_of_impacts, state_list, colors=['b','g','r']):
    
    # y here is 3D tensor with dimensions impacts x energies x variable values
    # t can be dimensional time or non-dimensional time
   
    fig, ax = plt.subplots()
    
    for c,v in enumerate(energy_list):
        legend_list = []
        for i in range(prob_no):
            ax.plot(t.detach().cpu().numpy(), y_pred[...,c,i].detach().cpu().numpy(), color=colors[i])
            ax.scatter(t.detach().cpu().numpy(), y_actual[...,c,i].detach().cpu().numpy(), color=colors[i], facecolors='none')
            legend_list.append(state_list[i]+' Predicted')
            legend_list.append(state_list[i]+' Simulation')
        ax.set(title=no_of_impacts[0]+' impacts '+v+' energy', xlabel='Time', ylabel='Probability')
        ax.legend(legend_list)
        fig.show()
            
            
def scale_tensor(xlist):
    # Scaling data: x' = 2*(x - min(x))/(max(x)-min(x))-1
    new_list = []
    minmax_list = []
    for item in xlist:
        xmin = torch.min(item)
        xmax = torch.max(item)
        item = (2 * (item - xmin) / (xmax - xmin))
        new_list.append(item)
        minmax_list.append([xmin, xmax])
    return new_list, minmax_list

def scale_single_val_tensor(num, nmax, nmin):
    return 2*(num-nmin)/(nmax-nmin)-1

def rescale_tensor(xlist, minmax_list):
    # x = 0.5*(x'+1)*(max(x)-min(x))+min(x)
    # The inputs are tensors this time around
    new_list = []
    for c,v in enumerate(xlist):
        xmin = minmax_list[c][0]
        xmax = minmax_list[c][1]
        item = 0.5*(v + torch.tensor(1)) * (xmax - xmin) + torch.tensor(xmin)
        new_list.append(item)
    return new_list
        
        
        
        
        
        
