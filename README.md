# Plasma-Etch-Simulation-and-Training
Plasma etch processes RIE and ALE can be modeled using a master equation approach, and leads to a system of coupled ODEs and a drift-diffusion PDE. The dependent variables are the state occupation probabilities of a species. The system of ODEs are characterized by the respective transition probabilities. 

To study the system, one needs to conduct molecular dynamics simulations of the system, whether that be ALE or RIE. This repository contain bash scripts and python scripts. The bash scripts can be used to conduct MD simulations using a C++ code in Savio cluster. These simulatios can be ensemble-DOE simulations as well. 

The python scripts are for three main purposes: MD data preprocessing, training models for the transition probabilities, and post-processing and model analysis.

This repo is for Si RIE.

Si RIE is carried out with F and Ar, at a flux ratio of 5, at different fluence and energy conditions. The starting configuration, of pure native Si, is dropped, and hence, in the simulation, the data starts from 10 fs. For training, the time is adjusted to start from 0. The rate of entry of F is constant, and is not solved with the master equations (analytical expression can be obtained for number of F atoms).

The diffusion coefficient is taken as 1e-19 m^2/s, and the value for the guess velocity is 1e-11 m/s. 
