#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import sys
import getopt
import numpy as np
import networkx as nx
import random
from scipy.sparse import triu

# RANDOM NETWORK

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

dt = 1
a_steps = 10 # Number of social activity upgrades at each time step

N_sims = 1000 # Number of simulation
N = 10000 # Number of individuals

delta = np.exp(-dt/10) # Discount factor

# Build the network 

prob = 14.7/N # 14.7 is the average degree

# BEHAVIOUR PARAMETERS

alpha = read_arguments(sys.argv)

# BIOLOGICAL PARAMETERS

mu = dt / 10.  # Recovery rate
beta_default = 3 * mu # Transmission rate

# CODE PARAMETERS

i0 = 0.01 # Fraction of initial infected nodes
# frac = 10/N # Only keep runs where the disease reaches this fraction of the population
t_max = round(1000/dt) # Max length of a simulation

def create_random_connected(N, prob, seed=0):
    G = nx.erdos_renyi_graph(N, prob, seed=seed)
    while not nx.is_connected(G):
        G = nx.erdos_renyi_graph(N, prob, seed=seed)
    return G

def get_prev_i(all_links, i_tuple: tuple, k):
    """
    Returns the prevalence of infected nodes in the neighbors of each node the network
    
    Parameters:
    all_links: array
        An array of all the links in the network,
        all_links = np.array(nx.adjacency_matrix(G).nonzero()).T
    i_tuple: tuple
        A tuple with all the infected nodes
    k: array
        An array with the degree of each node in the network
        k = np.array(G.degree)[:,1]
    """
    
    N = len(k)

    # only keep links where second node is infected
    infected_only = all_links[np.isin(all_links[:,1], i_tuple)]

    # count the number of infected neighbors for each node
    counts = np.zeros(N)
    has_i_neigh, counts[has_i_neigh] = np.unique(infected_only[:, 0], return_counts=True)

    # divide counts by degree
    prev = np.where(
        np.isin(range(N), has_i_neigh),
        counts / k,
        0
        )
    
    return prev

def get_avg_a(all_links, a, k):
    """
    Returns the average social activity (a) of the neighbors of each node in the network

    Parameters:
    all_links: array
        An array of all the links in the network,
        all_links = np.array(nx.adjacency_matrix(G).nonzero()).T
    a: array
        An array with the social activity of each node in the network
    k: array
        An array with the degree of each node in the network
        k = np.array(G.degree)[:,1]
    """

    N = len(k)

    # get social activity of each second node in the links
    a_neighs = a[all_links[:,1]]

    # sum the social activity of the neighbors for each node
    # the social activity of the neighbors is stored in a_neighs,
    # the number of neighbors is stored in k
    a_sum = np.bincount(all_links[:,0], weights=a_neighs)

    # divide by degree
    avg_a = a_sum / k

    return avg_a

def get_prev_s(all_links, i_tuple: tuple, r_tuple: tuple, k):
    """
    Returns the prevalence of susceptible nodes in the neighbors of each node the network

    Parameters:
    all_links: array
        An array of all the links in the network,
        all_links = np.array(nx.adjacency_matrix(G).nonzero()).T
    i_tuple: tuple
        A tuple with all the infected nodes
    r_tuple: tuple
        A tuple with all the recovered nodes
    k: array
        An array with the degree of each node in the network
        k = np.array(G.degree)[:,1]
    """

    N = len(k)
    
    s_tuple = np.setdiff1d(range(N), np.concatenate((i_tuple, r_tuple)))
    # only keep links where second node is susceptible
    susceptible_only = all_links[np.isin(all_links[:,1], s_tuple)]

    # count the number of susceptible neighbors for each node
    counts = np.zeros(N)
    has_s_neigh, counts[has_s_neigh] = np.unique(susceptible_only[:, 0], return_counts=True)

    # divide counts by degree
    prev = np.where(
        np.isin(range(N), has_s_neigh),
        counts / k,
        0
        )
    
    return prev

def simulate(N, prob, i0, t_max, beta_default, mu, alpha=0, seed=42, N_steps=10):

    G = create_random_connected(N, prob, seed=42)

    k = np.array(G.degree)[:,1]

    k_ave = 2*len(G.edges)/N
    beta = beta_default/k_ave

    # Init all nodes to S
    disease_status=np.array(["s"] * N)

    # Select i0*N nodes to be infected
    i_set = set(random.sample(range(0,N), max(1, round(N * i0))))
    r_set = set()

    # Init social activity (a) to 1
    a = np.ones(N)

    # non-zero elements are links, (symmetric)
    
    # All links (each link appears twice)
    all_links = np.array(nx.adjacency_matrix(G).nonzero()).T # 

    # only use the upper triangle (diagonal excluded)
    links_no_dupl = np.array(triu(nx.adjacency_matrix(G)).nonzero()).T

    result = {
        "s": [(N - len(i_set)) / N],
        "i": [len(i_set)/N],
        "r": [0]
        }

    for t in range(t_max):

        i_new = set()
        r_new = set()

        i_tuple = tuple(i_set)
        r_tuple = tuple(r_set)

        prev = get_prev_i(all_links, i_tuple, k) # prevalence
        sigma = get_prev_s(all_links, i_tuple, r_tuple, k) # prob of being S
        avg_a = get_avg_a(all_links, a, k) # avg social activity

        for _ in range(N_steps):
            a = 1 / (1 + alpha * delta * beta_default * sigma * prev * avg_a)

        # Remove links in which at least one node is recovered
        active_links = links_no_dupl[~np.isin(links_no_dupl, r_tuple).any(axis=1)]
        # Subset to links where only one node is infected
        active_links = active_links[np.sum(np.isin(active_links, i_tuple), axis=1) == 1]

        # Generate random numbers for each active link
        rand = np.random.rand(len(active_links))

        # save new infections
        i_thr = beta*a[active_links[:,0]] * a[active_links[:,1]]

        i_new = set(np.unique(active_links[rand < i_thr])) - i_set

        # Extract number of recoveries from a binomial distribution
        n_recoveries = np.random.binomial(len(i_set), mu)

        # Randomly select indices of nodes to recover
        rand = np.random.choice(len(i_set), n_recoveries, replace=False)

        r_new = set([i_tuple[i] for i in rand])

        # add new infections to infected nodes
        if i_new:
            i_set = i_set | i_new

        if r_new:
            # remove recoveries from infected nodes
            i_set = i_set - r_new
            # add new recoveries to recovered nodes
            r_set = r_set | r_new

        result["s"].append((N - len(i_set) - len(r_set)) / N)
        result["i"].append(len(i_set) / N)
        result["r"].append(len(r_set) / N)

        if len(i_set) == 0:
                # Repeat the last value of s, i and r until the end if the epidemics ends before tmax
                result["s"] = np.pad(result["s"], [(0, t_max+1-len(result["s"]))], mode='edge')
                result["i"] = np.pad(result["i"], [(0, t_max+1-len(result["i"]))], mode='constant') # pad 0
                result["r"] = np.pad(result["r"], [(0, t_max+1-len(result["r"]))], mode='edge')

                break 
        
    tt = np.linspace(0, t_max, t_max+1)
        
    return tt, result

# Create matrix "sims_matrix" with N_sims rows and t_max columns,
# each row represents the time-series of a single simulation

sims_matrix = np.zeros((N_sims, t_max+1))

N_keep = 0 # Number of runs in which i is over the threshold
r_inf = []  # Final attack rates list

for i, row in enumerate(sims_matrix):
    tt, result = simulate(
        N,
        prob,
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

path = "output/" + ("alpha={}/".format(alpha))

if not os.path.exists(path):
    os.makedirs(path)

np.save(path + "simulations", sims_matrix)
np.save(path + "r_inf", r_inf)
