#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import sys
import getopt
import numpy as np
import networkx as nx
import random

# SCALE FREE

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

def get_activity(k_min, k_max, gamma):
    #Estrae i degree da una legge di potenza con esponente gamma
    #k_min min degree
    #k_max max degree
    #gamma esponente
    
    y = random.uniform(0,1)
    
    e = gamma + 1.
    
    a = ( (k_max**e - k_min**e)*y + k_min**e )**(1./e)
    
    return a

def create_network():

    a=[]
    for i in range(N):
        act = get_activity(k_min, k_max, -gamma)
        a.append(int(round(act)))

    #we need the sum of the degree sequence to be even to properly run the configuration model
    if sum(a)%2==0:
        G = nx.configuration_model(a, seed=42)
    else:
        a[-1]+=1
        G = nx.configuration_model(a, seed=42)

    while not nx.is_connected(G):
        a=[]
        for i in range(N):
            act = get_activity(k_min, k_max, -gamma)
            a.append(int(round(act)))

        if sum(a)%2==0:
            G = nx.configuration_model(a, seed=42)
        else:
            a[-1]+=1
            G = nx.configuration_model(a, seed=42)

    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))

    #we need to store the disease status of each node
    G.disease_status={} #S=0, I=1, R=-1

    #we need to store theta for each node
    G.theta = {} #S=0, I=1, R=-1

    #we need to store the social activity of each node
    G.a={} #S=0, I=1, R=-1

    # We need to store the probability of being S for each node
    G.s={}

    # Local prevalence (rho in the paper)
    G.i={}

    return G

dt = 1
a_steps = 10 # Number of social activity upgrades at each time step

N_sims = 1000 # Number of simulations 1000
N = 10000 # Number of individuals 1e4 

delta = np.exp(-dt/10) # Discount factor

k_min = 5    # min degree
k_max = int(N**(0.5)) # max degree
gamma = 2.1  # power law exponent p_k = C*k^{-gamma}

# BEHAVIOUR PARAMETERS

alpha = read_arguments(sys.argv)

# BIOLOGICAL PARAMETERS

mu = dt / 10.  # Recovery rate
beta_default = 3 * mu # Transmission rate per single contact

# CODE PARAMETERS

i0 = 0.01 # Fraction of initial infected nodes
# frac = 10/N # Only keep runs where the disease reaches this fraction of the population
t_max = round(1000/dt) # Max length of a simulation

def simulate():
    # Simulates the epidemic. The change in social activity is only computed for infected nodes and
    # their neighbours

    G = create_network()
    k_ave = 2*len(G.edges)/N
    beta = beta_default/k_ave

    # Init
    infected_nodes = set(random.sample([node for node in G.nodes()], max(1, round(N * i0))))
    
    for i in G:
        # Set social activity to 1 and nodes to their initial status
        G.a[i] = 1
        if i in infected_nodes:
            G.disease_status[i] = "i"
        else:
            G.disease_status[i] = "s"

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
            G.s[i]     = sum([1 if G.disease_status[j] == "s" else 0 for j in G.neighbors(i)])/G.degree[i]
            G.i[i]     = sum([1 if G.disease_status[j] == "i" else 0 for j in G.neighbors(i)])/G.degree[i]
        
        # Update social activity a_step times to reach equilibrium
        for _ in range(a_steps):
          for i in effective_nodes: # Aggregate social activity to react to
                G.theta[i] = sum([G.a[j] for j in G.neighbors(i)])
          for i in effective_nodes: # Update all social activities
                G.a[i] = 1 / (1 + G.s[i]*G.i[i]*beta*G.theta[i]*delta*(alpha/dt))
        
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

path = "output/" + ("alpha={}/".format(alpha))

if not os.path.exists(path):
    os.makedirs(path)

np.save(path + "simulations", sims_matrix)
np.save(path + "r_inf", r_inf)

