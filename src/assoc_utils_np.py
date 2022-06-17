# corrupt_p01() and topk() from assoc_utils.py
# haven't been converted to numpy so they are missing 

import numpy as np
from scipy.ndimage import gaussian_filter1d
from numpy.random import rand
from numpy.random import randn 


# compute pattern correlations
# codebook = gbook/pbook/sbook
def correlation(codebook):
    return np.corrcoef(codebook, rowvar=False)


def extend_gbook(gbook, discretize):
    return np.repeat(gbook,discretize,axis=1)


def colvolve_1d(codebook, std):
    return gaussian_filter1d(codebook, std, mode="constant")


def cont_gbook(gbook, discretize=10, std=1):
    gbook = extend_gbook(gbook, discretize)
    gbook = colvolve_1d(gbook, std)
    return gbook


# generate modular grid code
def gen_gbook(lambdas, Ng, Npos):
    if len(lambdas) == 2:
        ginds = [0,lambdas[0]] 
    elif len(lambdas) == 3:
        ginds = [0,lambdas[0],lambdas[0]+lambdas[1]] 
    elif len(lambdas) == 4:
        ginds = [0,lambdas[0],lambdas[0]+lambdas[1],lambdas[0]+lambdas[1]+lambdas[2]]
    elif len(lambdas) == 5:
        ginds = [0,lambdas[0],lambdas[0]+lambdas[1],lambdas[0]+lambdas[1]+lambdas[2],
                 lambdas[0]+lambdas[1]+lambdas[2]+lambdas[3]]   
    gbook=np.zeros((Ng,Npos))
    for x in range(Npos):
        phis = np.mod(x,lambdas) 
        gbook[phis+ginds,x]=1    
    return gbook


def train_hopfield(pbook, Npatts):
    return (1/Npatts)*np.einsum('ijk, ilk->ijl', pbook[:,:,:Npatts], pbook[:,:,:Npatts])


def train_gcpc(pbook, gbook, Npatts):
    return (1/Npatts)*np.einsum('ij, klj -> kil', gbook[:,:Npatts], pbook[:,:,:Npatts])  
    

def pseudotrain_Wsp(sbook, ca1book, Npatts):
    ca1inv = np.linalg.pinv(ca1book[:, :, :Npatts])
    return np.einsum('ij, kjl -> kil', sbook[:,:Npatts], ca1inv[:,:Npatts,:]) 

def pseudotrain_Wps(ca1book, sbook, Npatts):
    sbookinv = np.linalg.pinv(sbook[:, :Npatts])
    return np.einsum('ij, kli -> klj', sbookinv[:Npatts,:], ca1book[:,:,:Npatts]) 
    

def pseudotrain_Wpp(ca1book, Npatts):
    ca1inv = np.linalg.pinv(ca1book[:, :, :Npatts])
    return np.einsum('ijk, ikl -> ijl', ca1book[:,:,:Npatts], ca1inv[:,:Npatts,:]) 

def pseudotrain_Wgp(ca1book, gbook, Npatts):
    ca1inv = np.linalg.pinv(ca1book[:, :, :Npatts])
    return np.einsum('ij, ljk -> lik', gbook[:,:Npatts], ca1inv[:,:Npatts,:]) 

def pseudotrain_Wgg(gbook, Npatts):
    gbookinv = np.linalg.pinv(gbook)
    return np.einsum('ij, jk -> ik', gbook[:,:Npatts], gbookinv[:Npatts,:])     


def corrupt_pmask(Np, pflip, ptrue, nruns):
    #flipmask = rand(nruns, Np)>(1-pflip)
    flipmask = rand(*ptrue.shape)>(1-pflip)
    ind = np.argwhere(flipmask == True)  # find indices of non zero elements 
    pinit = np.copy(ptrue) 
    return pinit, ind


# corrupts p when its -1/1 code
def corrupt_p(Np, pflip, ptrue, nruns):
    if pflip == 0:
        return ptrue
    pinit, ind = corrupt_pmask(Np, pflip, ptrue, nruns)
    pinit[ind[:,0], ind[:,1]] = -1*pinit[ind[:,0], ind[:,1]] 
    return pinit


def hopfield(pinit, ptrue, Niter, W):
    p = pinit
    for i in range (Niter):
        p = np.sign(W@p)
    return np.sum(np.abs(p-ptrue), axis=(1,2))/np.sum(np.abs(pinit-ptrue), axis=(1,2))


def gcpc(pinit, ptrue, Niter, Wgp, Wpg, gbook, lambdas, Np):
    m = len(lambdas)
    p = pinit
    for i in range(Niter):
        gin = Wgp@p;
        g = topk_binary(gin, m)         # non modular net
        p = np.sign(Wpg@g); 
    return np.sum(np.abs(p-ptrue), axis=(1,2))/np.sum(np.abs(pinit-ptrue), axis=(1,2))  #(2*Np)


# module wise nearest neighbor
def module_wise_NN(gin, gbook, lambdas):
    size = gin.shape
    g = np.zeros(size)               #size is (Ng, 1)
    i = 0
    for j in lambdas:
        gin_mod = gin[:, i:i+j]           # module subset of gin
        gbook_mod = gbook[i:i+j]
        g_mod = nearest_neighbor(gin_mod, gbook_mod)
        g[:, i:i+j, 0] = g_mod
        i = i+j
    return g  


# global nearest neighbor
def nearest_neighbor(gin, gbook):
    est = np.einsum('ijk, jl -> ikl', gin, gbook)
    maxm = np.amax(est, axis=2)       #(nruns,1)
    g = np.zeros((len(maxm), len(gbook)))
    for r in range(len(maxm)):
        a = np.argwhere(est[r] == maxm[r])
        idx = np.random.choice(a[:,1])
        g[r,:] = gbook[:,idx]; 
    return g


# return topk sparse code by setting 
# topk to 1 and all other values to zero
def topk_binary(gin, k):
    idx = np.argsort(gin, axis=1)
    idx = idx[:,-k:]
    idx = np.squeeze(idx)   # nruns x k
    g = np.zeros_like(gin) 
    nruns = gin.shape[0]   
    if k==1:
        g[np.arange(nruns),idx] = 1 
    else:               
        for i in range(k):
            g[np.arange(nruns),idx[:,i]] = 1
    return g


def default_model(lambdas, Ng, Np, pflip, Niter, Npos, gbook, Npatts_lst, nruns):
    # avg error over Npatts
    err_hop = -1*np.ones((len(Npatts_lst), nruns))
    err_gcpc = -1*np.ones((len(Npatts_lst), nruns))
    
    Wpg = randn(nruns, Np, Ng);           # fixed random gc-to-pc weights
    pbook = np.sign(np.einsum('ijk,kl->ijl', Wpg, gbook))  # (nruns, Np, Npos)

    k=0
    for Npatts in Npatts_lst:
        W = np.zeros((nruns, Np, Np));      # plastic pc-pc weights
        Wgp = np.zeros((nruns, Ng, Np));    # plastic pc-to-gc weights

        # Learning patterns 
        W = train_hopfield(pbook, Npatts)
        Wgp = train_gcpc(pbook, gbook, Npatts)

        # Testing
        sum_hop = 0
        sum_gcpc = 0 
        for x in range(Npatts): 
            ptrue = pbook[:,:,x,None]                       # true (noiseless) pc pattern
            pinit = corrupt_p(Np, pflip, ptrue, nruns)      # make corrupted pc pattern
            cleanup = hopfield(pinit, ptrue, Niter, W)      # pc-pc autoassociative cleanup  
            sum_hop += cleanup
            cleanup = gcpc(pinit, ptrue, Niter, Wgp, Wpg, gbook, lambdas)   # pc-gc autoassociative cleanup
            sum_gcpc += cleanup
        err_hop[k] = sum_hop/Npatts
        err_gcpc[k] = sum_gcpc/Npatts
        k += 1   

    return err_hop, err_gcpc    

def sparse_rand(nruns, Np, Ng, sparsity):
    W = -1*np.ones((nruns, Np,Ng))
    W[:, :, :sparsity] = 1
    #shuffles at each position in-place 
    # (random shuffling across neurons) 
    for j in range(nruns):
        for i in range(Np):
            np.random.shuffle(W[j,i,:])     
    return W


from src.theory_utils import *
def capacity_gcpc(lambdas, Ng, Np_lst, pflip, Niter, Npos, gbook, nruns):
    # avg error over Npatts
    Npatts_lst = np.arange(1,Npos+1)
    #Npatts_lst = [21]
    err_gcpc = -1*np.ones((len(Np_lst), len(Npatts_lst), nruns))

    l=0
    for Np in Np_lst:
        print("l = ",l)

        Wpg = randn(nruns, Np, Ng);           # fixed random gc-to-pc weights
        #Wpg = sparse_rand(nruns, Np, Ng, 3)
        pbook = np.sign(np.einsum('ijk,kl->ijl', Wpg, gbook))  # (nruns, Np, Npos)  

        k=0
        for Npatts in Npatts_lst:
            print("k = ",k)
            Wgp = np.zeros((nruns, Ng, Np));        # plastic pc-to-gc weights

            #Wgp = train_gcpc(pbook_grid, gbook_grid, np.prod(lambdas))
            Wgp = train_gcpc(pbook, gbook, Npatts)  # Training
            #Wgp = pseudotrain_Wgp(pbook, gbook, Npatts)

            # capacity for randomly sampled patterns
            #sampledpatt = np.random.choice(range(Npos), size=(Npatts), replace=False)

            # Testing
            sum_gcpc = 0 
            for x in range(Npatts): 
            #for x in range(Npos):
            #for x in sampledpatt:
                ptrue = pbook[:,:,x,None]                       # true (noiseless) pc pattern
                pinit = corrupt_p(Np, pflip, ptrue, nruns)      # make corrupted pc pattern
                cleanup = gcpc(pinit, ptrue, Niter, Wgp, Wpg, gbook, lambdas, Np)   # pc-gc autoassociative cleanup
                #if cleanup[0] > 0:
                    #print(cleanup[0], x)
                    #print(gbook[:,x,None].T)
                sum_gcpc += cleanup
            err_gcpc[l,k] = sum_gcpc/Npatts
            k += 1   
        l += 1    
    return err_gcpc                 


