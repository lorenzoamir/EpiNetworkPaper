#!/usr/bin/env python
# coding: utf-8

import os
from epieconlib import create_random_connected, simulate_sird
import sys
import argparse
import numpy as np
import networkx as nx
import random
from scipy.sparse import triu

# PARSE ARGUMENTS
parser = argparse.ArgumentParser(description='Run SIR simulations on a random network')
parser.add_argument('--alpha', type=float, help='Cost of infection', default=197)
parser.add_argument('--l-vs-g', type=float, help='Local vs global, 1=full local, 0=full global', required=True)
parser.add_argument('--outdir', type=str, help='Output directory', default='output/comparison/')
parser.add_argument('--beta', type=float, help='Transmission rate assuming dt = 1 day (automatically adjusted if dt!=1)', default=0.44285714285714284)
parser.add_argument('--kavg', type=float, help='Average degree of the network', default=14.7)
parser.add_argument('--mu', type=float, help='Recovery rate assuming dt = 1 day (is adjusted automatically if dt!=1)', default=0.14285714285714284)
parser.add_argument('--fatality', type=float, help='Fatality rate', default=0.0062)
parser.add_argument('--planninghorizon', type=float, help='Planning horizon in days, used to determine delta=exp(-dt/planning-horizon)', default=499.5)
parser.add_argument('--dt', type=float, help='Time step', default=1)
parser.add_argument('--activitysteps', type=int, help='Number of social activity upgrades at each time step', default=10)
parser.add_argument('--nsims', type=int, help='Number of simulations', default=100)
parser.add_argument('--nindividuals', type=int, help='Number of individuals', default=100000)
parser.add_argument('--tmax', type=int, help='Max length of a simulation in days', default=1000)
parser.add_argument('--seed', type=int, help='Random seed', default=42)
args = parser.parse_args()

# RANDOM NETWORK
dt = args.dt # Time step
a_steps = args.activitysteps # Number of social activity upgrades at each time step

N_sims = args.nsims # Number of simulations

# BEHAVIORAL PARAMETERS
delta = np.exp(-dt/args.planninghorizon)  # Discount factor
alpha = args.alpha  # Cost of infection
l_vs_g = args.l_vs_g

# NETWORK PARAMETERS
N = args.nindividuals # Number of individuals
prob = args.kavg/N # Probability of connection between two individuals

# BIOLOGICAL PARAMETERS
mu = args.mu * dt # Recovery rate
beta_default = args.beta * dt # Transmission rate
pi = args.fatality * dt # Fatality rate

# CODE PARAMETERS
# frac = 10/N # Only keep runs where the disease reaches this fraction of the population
t_max = round(args.tmax/dt) # Max length of a simulation

infected_matrix = np.zeros((N_sims, t_max+1))
recovered_matrix = np.zeros((N_sims, t_max+1))

# Init from Farboodi et al. calibrated on 51 deaths
#init = {
#    's': 0.9999223,
#    'i': 5.27e-5,
#    'r': 2.51e-5,
#}

# Our data has 52 deaths, so we recalibrate the initial conditions
init = {
    's': 0.99992074,
    'i': 5.369e-5,
    'r': 2.557e-5,
}

for i, (row_i, row_d) in enumerate(zip(infected_matrix, recovered_matrix)):
    G = create_random_connected(N, prob, seed=args.seed+i)

    tt, result = simulate_sird(
        G,
        init=init,
        t_max=t_max,
        beta_default=beta_default,
        mu=mu,
        pi=pi,
        alpha=alpha,
        delta=delta,
        l_vs_g=l_vs_g,
        seed=i,
        N_steps=a_steps
    )

    row_i[:len(result["i"])] = result["i"]
    row_d[:len(result["r"])] = result["r"]

# Create output directory if it does not exist
path = os.path.join(args.outdir, 'l_vs_g={}/'.format(l_vs_g))
os.makedirs(path, exist_ok=True)

np.save(os.path.join(path, "infected"), infected_matrix)
np.save(os.path.join(path, "recovered"), recovered_matrix)
