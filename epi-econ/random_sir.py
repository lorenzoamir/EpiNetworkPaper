#!/usr/bin/env python
# coding: utf-8

import os
from epieconlib import create_random_connected, simulate_sir
import sys
import argparse
import numpy as np
import networkx as nx
import random
from scipy.sparse import triu

# PARSE ARGUMENTS
parser = argparse.ArgumentParser(description='Run SIR simulations on a random network')
parser.add_argument('--alpha', type=float, help='Cost of infection', required=True)
parser.add_argument('--outdir', type=str, help='Output directory', default='.')
parser.add_argument('--beta', type=float, help='Transmission rate assuming dt = 1 day (automatically adjusted if dt!=1)', default=0.3)
parser.add_argument('--mu', type=float, help='Recovery rate assuming dt = 1 day (is adjusted automatically if dt!=1)', default=0.1)
parser.add_argument('--i0', type=float, help='Initial fraction of infected nodes', default=0.01)
parser.add_argument('--planninghorizon', type=float, help='Planning horizon in days, used to determine delta=exp(-dt/planning-horizon)', default=10)
parser.add_argument('--dt', type=float, help='Time step', default=1)
parser.add_argument('--activitysteps', type=int, help='Number of social activity upgrades at each time step', default=10)
parser.add_argument('--nsims', type=int, help='Number of simulations', default=100)
parser.add_argument('--nindividuals', type=int, help='Number of individuals', default=10000)
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

# NETWORK PARAMETERS
N = args.nindividuals # Number of individuals
prob = 14.7/N # 14.7 is the average degree

# BIOLOGICAL PARAMETERS
mu = args.mu * dt # Recovery rate
beta_default = args.beta * dt # Transmission rate

# CODE PARAMETERS
i0 = args.i0 # Initial fraction of infected nodes
# frac = 10/N # Only keep runs where the disease reaches this fraction of the population
t_max = round(args.tmax/dt) # Max length of a simulation

# Create matrix "sims_matrix" with N_sims rows and t_max columns,
# each row represents the time-series of a single simulation

sims_matrix = np.zeros((N_sims, t_max+1))

N_keep = 0 # Number of runs in which i is over the threshold
r_inf = []  # Final attack rates list

for i, row in enumerate(sims_matrix):
    G = create_random_connected(N, prob, seed=args.seed+i)

    tt, result = simulate_sir(
        G,
        i0,
        t_max,
        beta_default,
        mu,
        alpha=alpha,
        seed=i,
        N_steps=a_steps
    )

#    if(result["r"][-1] >= frac): # Only keep runs where the desease reaches a significant fraction of the pupulation
    row[:] = result["i"][:]
    r_inf.append(result["r"][-1])

r_inf = np.array(r_inf) # From list to np.array

# Create output directory if it does not exist
path = os.path.join(args.outdir, "alpha={}/".format(alpha))
os.makedirs(path, exist_ok=True)

if not os.path.exists(path):
    os.makedirs(path)

np.save(path + "simulations", sims_matrix)
np.save(path + "r_inf", r_inf)
