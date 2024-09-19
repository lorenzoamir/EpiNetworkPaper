import numpy as np
import networkx as nx
import random
from scipy.sparse import triu

# ----- NETWORK CRATION -----
def create_random_connected(N, prob, seed=0):
    """
    Create a random connected network with N nodes and a given probability of connection

    parameters:
    N: int
        Number of nodes
    prob: float
    """
    G = nx.erdos_renyi_graph(N, prob, seed=seed)
        
    while not nx.is_connected(G):
        # Update seed
        seed += 1
        G = nx.erdos_renyi_graph(N, prob, seed=seed)
    return G

def create_scalefree(N, k_min, k_max, gamma, seed=0):
    """
    Create a scale-free network with N nodes and a power law degree distribution

    parameters:
    N: number of nodes
    k_min: minimum degree
    k_max: maximum degree
    gamma: exponent
    """

    a=[]
    for i in range(N):
        act = get_activity(k_min, k_max, -gamma)
        a.append(int(round(act)))

    #we need the sum of the degree sequence to be even to properly run the configuration model
    if sum(a)%2==0:
        G = nx.configuration_model(a, seed=seed)
    else:
        a[-1]+=1
        G = nx.configuration_model(a, seed=seed)

    while not nx.is_connected(G):
        seed += 1
        a=[]
        for i in range(N):
            act = get_activity(k_min, k_max, -gamma)
            a.append(int(round(act)))

        if sum(a)%2==0:
            G = nx.configuration_model(a, seed=seed)
        else:
            a[-1]+=1
            G = nx.configuration_model(a, seed=seed)

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

    return G

def get_activity(k_min, k_max, gamma):
    """
    Function to extact degrees from a power law distribution with exponent gamma

    Parameters:
        k_min: minimum degree
        k_max: maximum degree
        gamma: exponent
    """

    y = random.uniform(0,1)
    
    e = gamma + 1.
    
    a = ( (k_max**e - k_min**e)*y + k_min**e )**(1./e)
    
    return a

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

def simulate_sir(G, i0, t_max, beta_default, mu, alpha=0, delta=0.9, seed=42, N_steps=10):
    
    N = len(G.nodes)
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

        for _ in range(N_steps):
            avg_a = get_avg_a(all_links, a, k) # avg social activity
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

def simulate_sis(G, i0, t_max, beta_default, mu, alpha=0, delta=0.9, seed=42, N_steps=10):
    
    N = len(G.nodes)
    k = np.array(G.degree)[:,1]

    k_ave = 2*len(G.edges)/N
    beta = beta_default/k_ave

    # Init all nodes to S
    disease_status=np.array(["s"] * N)

    # Select i0*N nodes to be infected
    i_set = set(random.sample(range(0,N), max(1, round(N * i0))))
    s_set = set(np.setdiff1d(range(N), i_set)) # all other nodes are susceptible
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

    r_tuple = tuple() # no recovered nodes

    for t in range(t_max):

        s_new = set()
        i_new = set()
        r_new = set()

        s_tuple = tuple(s_set)
        i_tuple = tuple(i_set)
        r_tuple = tuple(r_set)

        prev = get_prev_i(all_links, i_tuple, k) # prevalence
        sigma = get_prev_s(all_links, i_tuple, r_tuple, k) # prob of being S

        for _ in range(N_steps):
            avg_a = get_avg_a(all_links, a, k) # avg social activity
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

        s_new = set([i_tuple[i] for i in rand])

        # add new infections to infected nodes
        if i_new:
            s_set = s_set - i_new
            i_set = i_set | i_new

        # new recoveries
        if s_new:
            i_set = i_set - s_new
            s_set = s_set | s_new

        result["s"].append(len(s_set) / N)
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

