#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import sys
import getopt
import numpy as np
import networkx as nx
import random

def read_arguments(argv):
    arg_alpha = ""
    arg_help = "{0} -i <input> \n The input is the value of the parameter alpha".format(argv[0])
    
    try:
        opts, args = getopt.getopt(argv[1:], "hi:u:o:", ["help", "input=", 
        "user=", "output="])
    except:
        print(arg_help)
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(arg_help)  # print the help message
            sys.exit(2)
        elif opt in ("-i", "--input"):
            arg_alpha = float(arg)
            return arg_alpha

# In[2]:

N_sims = 1000 # Number of simulations
N = 100000 # Number of individuals

delta = 0.9 # Discount factor

# Build the network 

prob = 17.38/N # 17.38 is the average degree

# BEHAVIOUR PARAMETERS

alpha = read_arguments(sys.argv)

# BIOLOGICAL PARAMETERS

mu = 0.1  # Recovery rate
beta_default = 0.3 # Transmission rate

# CODE PARAMETERS

i0 = 0.01 # Fraction of initial infected nodes
# frac = 10/N # Only keep runs where the disease reaches this fraction of the population
t_max = 1000 # Max length of a simulation

# In[5]:

# In[6]:

def simulate():
    # Simulates the epidemic. The change in social activity is only computed for infected nodes and
    # their neighbours
    
    G = nx.fast_gnp_random_graph(N, prob, seed=42)

    while not nx.is_connected(G):
        G = nx.fast_gnp_random_graph(N, prob, seed=42)


    # Init
    infected_nodes = set(random.sample([node for node in G.nodes()], max(1, round(N * i0))))
    
    # We need to store the disease status of each node
    G.disease_status={}

    # We need to store theta for each node
    G.theta = {}

    # We need to store the social activity of each node
    G.a={}
    
    for i in G:
        # Set social activity to 1 and nodes to their initial status
        G.a[i] = 1
        if i in infected_nodes:
            G.disease_status[i] = "i"
        else:
            G.disease_status[i] = "s"

    k_ave = 2*len(G.edges)/N
    beta = beta_default/k_ave

    result = {"s": [1 - len(infected_nodes)/N],
              "i": [len(infected_nodes)/N],
              "r": [0]}
    
    t = 0
    
    effective_nodes = set() # Only I nodes and their neighbours are relevant

    for i in infected_nodes:
        if i not in effective_nodes:
            effective_nodes.add(i)
        for j in G.neighbors(i):
            if j not in effective_nodes:
                effective_nodes.add(j)

    for t in range(0, t_max):

        dSI = 0
        dIR = 0
        
        # Update theta
        for i in effective_nodes:
            G.theta[i] = sum([G.a[j] if G.disease_status[j] == "i" else 0 for j in G.neighbors(i)])
        
        # Update social activity
        for i in effective_nodes:
            G.a[i] = 1 / (1 + beta*G.theta[i]*delta*alpha)
        
        infected_add  = set() # Will be added to infected
        effective_add = set() # Will be added to effective
        
        for i in infected_nodes:
            for j in G.neighbors(i):
                if G.disease_status[j]=="s":
                    p=np.random.random()
                    if p<beta*G.a[i]*G.a[j]:
                        G.disease_status[j]="i" # S -> I
                        infected_add.add(j)
                        effective_add.add(j)                       
                        dSI += 1
                        for k in G.neighbors(j): # Add new I node's neighbours to effective
                            if G.disease_status[k] == "s": # Infected nodes are already effective
                                effective_add.add(k)       # recovered nodes should not be effective 

        infected_remove  = set() # Will be removed from infected
        effective_remove = set() # Will be removed from effective
        
        for i in infected_nodes:
            p = np.random.random()
            if p<mu:
                G.disease_status[i]="r" # I -> R
                infected_remove.add(i) # remove from infected
                effective_remove.add(i) # remove from effected
                dIR += 1
                for j in G.neighbors(i):
                    if j in effective_nodes and G.disease_status[j] == "s":
                        if "i" not in [G.disease_status[k] for k in G.neighbors(j)]:
                            effective_remove.add(j)
                            # If the list of the health statuses of the neighbours (k) of the neighbour (j)
                            # of the recovered node (i) doesn't contain any infected node, then the node j
                            # can be removed from the effective nodes.
                            # We also need to check that the node is in the S state, otherwise it could be
                            # an I node (which should be effective) even if it doesn't have I neighbours
                
        for i in infected_add:
            infected_nodes.add(i)
        for i in infected_remove:
            infected_nodes.remove(i)
            
        for i in effective_add:
            if i not in effective_nodes:
                effective_nodes.add(i)
        for i in effective_remove:
                effective_nodes.remove(i)

        result["s"].append(result["s"][-1] - dSI/N)
        result["i"].append(result["i"][-1] + dSI/N - dIR/N)
        result["r"].append(result["r"][-1] + dIR/N)

        t+=1
    
        if len(infected_nodes) == 0:
            # Repeat the last value of s, i and r until the end if the epidemics ends before tmax
            result["s"] = np.pad(result["s"], [(0, t_max+1-len(result["s"]))], mode='edge')
            result["i"] = np.pad(result["i"], [(0, t_max+1-len(result["i"]))], mode='constant') # pad 0
            result["r"] = np.pad(result["r"], [(0, t_max+1-len(result["r"]))], mode='edge')

            break 
    
    tt = np.linspace(0, t_max, t_max+1)
    
    return tt, result


# In[7]:


# Let's create a matrix "sims_matrix" with N_sims rows and t_max columns,
# each row represents the time-series of a single simulation

sims_matrix = np.zeros((N_sims, t_max+1))

N_keep = 0 # Number of runs in which i is over the threshold
r_inf = []  # Final attack rates list

for i, row in enumerate(sims_matrix):
    tt, result = simulate()

#    if(result["r"][-1] >= frac): # Only keep runs where the desease reaches a significant fraction of the pupulation
    row[:] = result["i"][:]
    r_inf.append(result["r"][-1])

r_inf = np.array(r_inf) # From list to np.array

# We are keeping rows for which it's false that all elements are 0 using the complement operator "~"
# sims_matrix = sims_matrix[~np.all(sims_matrix == 0, axis=1)]

path = "output/" + ("alpha={}/".format(alpha))

if not os.path.exists(path):
    os.makedirs(path)

np.save(path + "simulations", sims_matrix)
np.save(path + "r_inf", r_inf)

