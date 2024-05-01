# Plasma-Etch-Simulation-and-Training
Plasma etch processes RIE and ALE can be modeled using a master equation approach, and leads to a system of coupled ODEs and a drift-diffusion PDE. The dependent variables are the state occupation probabilities of a species. The system of ODEs are characterized by the respective transition probabilities. 

To study the system, one needs to conduct molecular dynamics simulations of the system, whether that be ALE or RIE. This repository contain bash scripts and python scripts. The bash scripts can be used to conduct MD simulations using a C++ code in Savio cluster. These simulatios can be ensemble-DOE simulations as well. 

The python scripts are for three main purposes: MD data preprocessing, training models for the transition probabilities, and post-processing and model analysis.
