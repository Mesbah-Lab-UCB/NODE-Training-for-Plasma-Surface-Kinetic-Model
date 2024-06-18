#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 14:06:45 2023

@author: shoubhaniknath
"""

""" 
    Python module. This module takes in file paths to cfg files and txt files, and timestep as inputs.
    The methods can output tuples of Cl atom IDs, dataframe of bond parameters from txt files, dataframe
    of bond parameters for Cl atoms, occupation number of states of Cl atoms
"""


import pandas as pd
import numpy as np

class Atom_Properties:
    def __init__(self,cfg_path,txt_path,timestep):
        self.cfg_path = cfg_path                                                    #path to .cfg file
        self.txt_path = txt_path                                                    #path to .txt file containing bond parameters
        self.dt = timestep                                                          #takes in the current timestep
        
        #Boolean variables to ensure execution of dependent methods
        self.atom_id_exec = False
        self.bond_param_df_gen_exec = False
        self.cl_atom_info_exec = False
    
    halogen_atomic_num = 9
    '---Method to create tuples of Si and Cl atom IDs from .cfg file---'
    def atom_id(self):
        
        file_path = self.cfg_path 
        
        # Read the file line by line and filter out unwanted lines
        lines = []
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith('#N'):
                    n_value = int(line.split()[1])                                  #extract the total number of atoms
                elif not line.startswith('#') and not line.startswith('%'):         #ignore other lines that start with # and %
                    line = line.replace('<', '').replace('>', '')                   #remove the '<' and '>' characters
                    lines.append(line)                                              

        # Process the filtered lines and create a DataFrame
        data = []
        for line in lines:
            values = line.split()
            data.append([int(values[0]), int(values[1]), values[4]])                          #extract the atom ID and atomic number
        
        df1 = pd.DataFrame(data, columns=['Atom ID','Atomic Number','Z-Coordinate'])
        cl_atom_dict = dict(zip(df1[df1['Atomic Number'] == self.halogen_atomic_num]['Atom ID'],\
                                df1[df1['Atomic Number'] == self.halogen_atomic_num]['Z-Coordinate']))              #tuple for cl atom IDs
        si_atom_id = tuple(df1[df1['Atomic Number'] == 14]['Atom ID'])              #tuple for si atom IDs
        
        self.atom_id_exec = True                                                    #method has been executed
        return cl_atom_dict, si_atom_id
    
    '---Method to create a dataframe of bond parameters from .txt file---'
    def bond_param_df_gen(self):
        df = pd.read_csv(self.txt_path, header=None)
        df.columns = ['Atom i','Atom j','Force','Energy','Cutoff','Bond Order']     #this is the format in the generated .txt files
        self.bond_param_df_gen_exec = True                                          #method has been executed
        return df
    
    '---Method to create a dataframe of average bond quantities for Cl atoms---'
    def cl_atom_info(self):
        if not self.atom_id_exec:                                                   #ensures that if the method has been executed before, then it does not execute it all over again. Reduces computation
            cl_atom_id, si_atom_id = self.atom_id()
        df = self.bond_param_df_gen()                                               #dataframe of bond parameters from a particular .txt file
        names = list(df.columns)    
        names = names[2:]                                                           #ignores the column names of Atom i and Atom j
        columns = [i for i in names]
        columns.insert(0,'Cl atom id')                                              #list of column names for cl atom bond info
        if len(cl_atom_id)==0:                                                      #useful for the first configuration with no Cl atoms, or if there are no Cl atoms in any configuration
            df_blist = pd.DataFrame(columns = ['Cl atom id','Coordination #','Avg Force','Avg Energy','Avg Cutoff','Avg BO'])
        else:
            df_blist = pd.DataFrame()
        position_dict = {}
        key_list = list(cl_atom_id.keys())
        for c,i in enumerate(cl_atom_id):                                           #empty if no Cl atoms are present
            
            # tuple of indices where the atom ID appears in either Atom i or Atom j
            # This is used to count the coordination number
            mask = np.where((df['Atom i']==i) | (df['Atom j']==i))                  #returns a tuple of indices where the condition is true  
            indices = df.index[mask]                                                #creates a series of the index values from above  
            df_temp = df[names].iloc[mask[0]].reset_index(drop = True)              #creates a temporary dataframe which stores the relevant values of the Cl atoms  
            ser = pd.Series(np.ones(len(mask[0]))*i, name='Cl atom id')             #will be used to add the atom id to the dataframe  
            
            # Coordination number for that atom ID
            coord_num = len(mask[0])                                                #total instances = total number of si atoms bonded to it = coordination number
            
            # Calculation of average quantities
            avg_force = df_temp['Force'].mean()
            avg_energy = df_temp['Energy'].mean()
            avg_cutoff = df_temp['Cutoff'].mean()
            avg_bond_order = df_temp['Bond Order'].mean()
            zposition = cl_atom_id[i]
            
            if coord_num > 2:
                position_dict[key_list[c]] = zposition
            # Stores average quantities to a dictionary
            val_dict = {'Cl atom id':i,'Coordination #':coord_num,'Avg Force':avg_force,'Avg Energy':avg_energy,'Avg Cutoff':avg_cutoff,'Avg BO':avg_bond_order}
            df_temp = pd.DataFrame(data=val_dict,index=[0])
            
            df_blist = pd.concat([df_blist,df_temp]).reset_index(drop=True)
        
        self.cl_atom_info_exec = True
        return df_blist, position_dict                                                               #returns a dataframe of the average quantities for each Cl atom
    
    '---Method to calculate population of states of Cl atoms---'
    def cl_occupation(self):
        if not self.cl_atom_info_exec:                                                #ensures method execution if not executed before  
            df_blist, position_dict = self.cl_atom_info()
        
        surface_num = len(df_blist[df_blist['Coordination #']<=2])                    #number of surface atoms of Cl  
        bulk_num = len(df_blist[df_blist['Coordination #']>2])                        #number of bulk atoms of Cl  
        total_num = surface_num+bulk_num
        time = self.dt
        
        data = np.array([[time,surface_num,bulk_num,total_num]])
        cl_occupation_df = pd.DataFrame(data, columns = ['Time','Surface','Mixed Layer','Total'])
        cl_position_df = pd.DataFrame(np.array([[time, position_dict]]), columns = ['Time','Position Dict'])
        cl_position_df.set_index(cl_occupation_df.index, inplace=True)
        return cl_occupation_df, cl_position_df
        
        