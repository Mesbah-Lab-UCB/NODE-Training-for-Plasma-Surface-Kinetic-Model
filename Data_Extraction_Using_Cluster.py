#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Apr 12 16:59:33 2024

@author: shoubhaniknath
   Python script to extract raw data from MD simulation of Si ALE. This script generates .csv files
   of the occupation number and Cl position, from ENSEMBLE-DOE simulations.

   This script is designed to be used in conjunction with a bash file to run in parallel in Savio cluster
   Extracting data from the login node takes too much time. The associated bash file to run in Savio cluster 
   is MD_Data_Preprocessing.sh
"""

import pandas as pd
import numpy as np
import time
import sys
sys.path.append('/Users/shoubhaniknath/my_modules/')
from Atom_Properties import Atom_Properties
import os
from collections import Counter

start_time = time.time()
fluence = 100
energy = 49
number = 2
process = 'ALE'

def Arrange_CFG_RIE(cfg_list):
    cfg_length = list(range(len(cfg_list)))
    cfg_list = [f'Impact_{i}_Cycle_1.cfg' for i in cfg_length]
    return cfg_list
    
def RIE_F_Total(df, flux_ratio):
    counter = flux_ratio
    Ftot_count = np.arange(0, len(df))
    while counter<len(df):
        Ftot_count[counter] = Ftot_count[counter-1]
        counter = counter + flux_ratio
    df['Total'] = Ftot_count
    return df

def Probability_calc(df):
    df.drop(0, inplace=True)
    surface = np.array(matrix['Surface'])
    mixed_layer = np.array(matrix['Mixed Layer'])
    total = np.array(matrix['Total'])
    surface_prob = surface/total
    mixed_layer_prob = mixed_layer/total
    df['Surface Prob'] = surface_prob
    df['Mixed Layer Prob'] = mixed_layer_prob
    return df

folder_path = '/Users/shoubhaniknath/Documents/Rit PhD/Surface Kinetic Model/Si ALE with Cl-Ar/States of Si/Data '+str(fluence)+' impacts energy '+str(energy)+' number '+str(number)

if process == 'ALE':
    etch_product_file = folder_path + '/etch_products.txt'

    # Read the file line by line and filter out unwanted lines
    lines = []; 

    run_list = []; si_df = pd.DataFrame(columns=['Time','Si_Sputtered'])
    # We already know the total number of Si atoms that we are starting with
    with open(etch_product_file, 'r') as file:
       for line in file:
           line = line.strip()
           if '-run ' in line:
               # Time step                                                     
               run = line.split('run')[-1].split()[0].replace('-','')
               si_df.loc[len(si_df)] = [int(run)*10,0]
           elif line.startswith('* Cluster'):
               line = line.replace('* Cluster of','').replace('sputtered','')
               product = line.split()
               lines.append(product[0])
               run_list.append(int(run))
               
    products = list(Counter(lines))
    si_list = [0]; si_count=0

    for c,v in enumerate(lines):
        si_count=0
        if v.startswith('Si'):
            char = []
            char[:] = v
            if len(char) >2 and char[2].isnumeric():
                si_count += int(char[2])
                si_list.append(si_count + si_list[-1])
                si_df['Si_Sputtered'].iloc[run_list[c]:] = si_list[-1]
            elif len(si_list)==0:
                si_count += 1
                si_list.append(si_count)
                si_df['Si_Sputtered'].iloc[run_list[c]:] = si_list[-1]
            else:
                si_count += 1
                si_list.append(si_count + si_list[-1])
                si_df['Si_Sputtered'].iloc[run_list[c]:] = si_list[-1]
    # with open(etch_product_file, 'r') as file:
    #     for line in file:
    #         line = line.strip()
    #         if 'run' in line:
    #             # Time step
    #             run = line.split('run')[-1].split()[0].replace('-','')
    #         elif line.startswith('* Cluster'):
    #             line = line.replace('* Cluster of','').replace('sputtered','')
    #             product = line.split()
    #             lines.append(product[0]) 
    #             run_list.append(int(run)*10)
    
                
    # products = list(Counter(lines))
    # si_list = []; si_count=0
    
    # for item in lines:
    #     if item.startswith('Si'):
    #         char = []
    #         char[:] = item
    #         if len(char) >2 and char[2].isnumeric():
    #             si_count += int(char[2])
    #             si_list.append(si_count)
    #         else:
    #             si_count += 1
    #             si_list.append(si_count)


si_sputtered_dict = dict(zip(run_list, si_list))

os.chdir(folder_path)
# Get a list of all .txt files in the folder                              
txt_file_list = [file for file in os.listdir(folder_path) if file.endswith('.txt')]
cfg_file_list = [file for file in os.listdir(folder_path) if file.endswith('.cfg')]

# Remove the etch_products file from the list
txt_file_list.remove('etch_products.txt')
txt_file_list = sorted(txt_file_list, key=lambda x: int(x.split('.')[0]));

if process == 'ALE':
    cfg_file_list.sort()
elif process == 'RIE':
    cfg_file_list = Arrange_CFG_RIE(cfg_file_list)


dt = 10                              #time step of MD (femtoseconds)      
simtime = 0

cl_impact = 0                        #total Cl impacts in each cycle
ar_impact = fluence                          #total Ar imapcts in each cycle                                                                              
snapshots = cl_impact+ar_impact+1       #total number of snapshots in the file. The 1 refers to initial config  

# Raise exceptions if there is mismatch between number of files                                                                                 
#if len(txt_file_list)!=snapshots:
#    raise Exception('No. of .txt files not equal to no. of snapshots. Possible solution is to check variables cl_impact and ar_impact.')
#elif len(cfg_file_list)!=snapshots:
#    raise Exception('No. of .cfg files not equal to no. of snapshots. Possible solution is to check variables cl_impact and ar_impact.')
matrix = pd.DataFrame(columns = ['Time','Surface','Mixed Layer','Total'])
df = pd.DataFrame(columns = ['Time','Position Dict'])

'---This loop, at every instance, should pass the cfg file and txt file to the module. Then, it should---'
'---collate the results, and generate a dataframe for the occupation number and probability---'

for j in range(len(txt_file_list)):                                            #loop to run over each file                                      
    if os.path.getsize(txt_file_list[j])==0:
        raise Exception(f'File number {j}.txt in folder of {fluence} impacts {energy} energy number {number} is empty')
    aout = Atom_Properties(cfg_file_list[j], txt_file_list[j], simtime)        #instance of object of Atom_Properties

    # Are you using F or Cl
    aout.halogen_atomic_num = 17
    simtime += dt
    temp_occupation_df, temp_position_df = aout.cl_occupation()
    matrix = pd.concat([matrix,temp_occupation_df],axis=0)
    df = pd.concat([df,temp_position_df], axis=0)

 #Calculating the probabilites from the occupation number                                                                                        
matrix = matrix.reset_index(drop=True)
df = df.reset_index(drop=True)
#matrix['Surface Prob'] = matrix['Surface']/matrix['Total'].iloc[-ar_impact-1] #This can be used instead, to ensure that the normalizing numberis taken from just before Ar impacts start                                                                                                                 
# Reposition column for Mixed Layer Probability
#ml_prob = matrix.pop('Mixed Layer Prob')
#matrix.insert(len(matrix.columns), 'Mixed Layer Prob', ml_prob)
                           
if process == 'ALE':
    # Provided no fixes have been added
    matrix['Time'] = range(0,snapshots*dt,dt)
    matrix['Surface Prob'] = matrix['Surface']/matrix['Total'].max()
    matrix['Mixed Layer Prob'] = matrix['Mixed Layer']/matrix['Total'].max()
    
elif process == 'RIE':
    flux_ratio = 5
    matrix = RIE_F_Total(matrix, flux_ratio)
    matrix = Probability_calc(matrix)

# For RIE (if fixes have been added)
# matrix['Time'] = range(0,len(matrix.Bulk)*dt,dt)

# For ALE (if no fixes have been added)
#
# Total Si at start of simulation (get this from the temp_000000-000.cfg file

NSi = 1280.0
matrix['Si Bulk Prob'] = 0
matrix['Si Bulk Prob'] = matrix['Time'].map(si_sputtered_dict).fillna(matrix['Si Bulk Prob'])
matrix['Si Bulk Prob'] = matrix['Si Bulk Prob'] / NSi

# matrix.to_csv('/Users/shoubhaniknath/Documents/Rit PhD/Surface Kinetic Model/Si ALE with Cl-Ar/States of Si/Occupation_Number_Impacts_'+str(fluence)+'_Energy_'+str(energy)+'_Number_'+str(number)+'.csv', index=False)
# df.to_csv('/Users/shoubhaniknath/Documents/Rit PhD/Surface Kinetic Model/Si ALE with Cl-Ar/States of Si/Cl_position_matrix_Impacts_'+str(fluence)+'_Energy_'+str(energy)+'_Number_'+str(number)+'.csv', index=False)

with open("time_log","a") as f:
    f.write(f'Run time = {time.time() - start_time}')

