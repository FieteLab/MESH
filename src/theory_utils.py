import numpy as np
from numpy.random import shuffle 
from src.assoc_utils_np import correlation
import matplotlib.pyplot as plt
import itertools
import math


# rearranges nCk gbook to have original 
# prod(lambdas) gbook at the beginning
def rearrange_gbook(gbook_grid, gbook_nck):
    N = gbook_grid.shape[1]
    for i in range(N):  
        idx = np.argmax(gbook_grid.T[i]@gbook_nck)
        swap = np.copy(gbook_nck[:,i])
        gbook_nck[:,i] = gbook_nck[:,idx]
        gbook_nck[:,idx] = swap
    return gbook_nck


# computes rank of a codebook
def rank(codebook):
    return np.linalg.matrix_rank(codebook)


# computes rank of subsets of codebook
# all consecutive subsets until idx
def rank_sub(codebook, idx=25):
    rank_part = np.zeros((idx))
    for i in range(idx): 
      rank_part[i] = np.linalg.matrix_rank(gbook[:,:i+1])
    # plt.figure()
    # plt.plot(range(idx), rank_part, 'ro--')
    # plt.xlabel("patterns in grid code book", fontsize=16) 
    # plt.ylabel("rank of grid code book", fontsize=16) 
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.savefig(f"aftercosyne_plots/gbook_rank_zoomed_lambdas={lambdas}")
    # plt.savefig(f"aftercosyne_plots/gbook_rank_zoomed_lambdas={lambdas}.svg", format="svg")
    # plt.show()
    return rank_part


# shuffles grid code at each position 
# in-place (random shuffling across neurons)         
def shuffle_gcode(gbook, Npos):
    for i in range(Npos):
        shuffle(gbook[:,i]) 
    return gbook


# shuffles gbook across positions
def shuffle_gbook(gbook):   
    gbook = np.random.permutation(np.transpose(gbook))   # need to transpose since permutation only shuffles along first index
    gbook = np.transpose(gbook)                # transpose back 
    return gbook


# scale matrix values to lie between 0 and 1
def normmat(W):
    normW = W - np.min(W)
    normW = normW / np.max(normW)
    return normW


def tran_var(Np_lst, lambdas, gbook, Ng):
    runs = 50
    mean_lst = []
    std_lst = []
    for Np in Np_lst:
        meanvar_r = []
        for j in range(runs):
            Wpg = np.random.randn(Np, Ng);           # fixed random gc-to-pc weights
            pbook = np.sign(np.einsum('jk,kl->jl', Wpg, gbook))  # (Np, Npos)
            corr = correlation(pbook)
            Npos = len(corr)

            lst = []
            for i in range(Npos):
                var = np.var(np.diagonal(corr, offset=i))
                lst.append(var)
            mean = np.min(lst)
            #mean = np.mean(lst)
            meanvar_r.append(mean) 
        mean_lst.append(np.mean(meanvar_r))
        std_lst.append(np.std(meanvar_r))
    return mean_lst, std_lst   


def nCr(n,r):
    f = math.factorial
    return f(n) // f(r) // f(n-r)


# print kbits(4, 3)
# Output: [1110, 1101, 1011, 0111]  array of shape (Ng,Npos)
def kbits(n, k):
    result = []
    for bits in itertools.combinations(range(n), k):
        s = [0.] * n
        for bit in bits:
            s[bit] = 1.
        result.append(s)
    return np.array(result).T  
    

