import numpy as np
from scipy.ndimage import gaussian_filter1d
from numpy.random import rand
from numpy.random import randn 


# global nearest neighbor
def nearest_neighbor(gin, gbook):
    est = np.transpose(gin)@gbook; 
    a = np.where(est[0,:]==max(est[0,:]))
    #print("Nearest neighbor: ", a)
    idx = np.random.choice(a[0])
    g = gbook[:,idx]; 
    return g


# module wise nearest neighbor
def module_wise_NN(gin, gbook, lambdas):
    size = gin.shape
    g = np.zeros(size)               #size is (Ng, 1)
    i = 0
    for j in lambdas:
        gin_mod = gin[i:i+j]           # module subset of gin
        gbook_mod = gbook[i:i+j]
        g_mod = nearest_neighbor(gin_mod, gbook_mod)
        g[i:i+j, 0] = g_mod
        i = i+j
    return g    


# return topk sparse code by setting 
# all other values to zero
def topk(gin, k):
    idx = np.argsort(gin, axis=0)
    idx = idx[-k:]
    size = gin.shape
    g = np.zeros(size) 
    g[idx] = gin[idx]
    return g


# return topk sparse code by setting 
# topk to 1 and all other values to zero
def topk_binary(gin, k):
    idx = np.argsort(gin, axis=0)
    print(idx)
    idx = idx[-k:]
    size = gin.shape
    g = np.zeros(size) 
    g[idx] = 1
    return g


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
    ginds = [0,lambdas[0],lambdas[0]+lambdas[1]]; 
    gbook=np.zeros((Ng,Npos))
    for x in range(Npos):
        phis = np.mod(x,lambdas) 
        gbook[phis+ginds,x]=1 
    return gbook


def corrupt_pmask(Np, pflip, ptrue):
    flipmask = rand(Np,1)>(1-pflip)
    ind = np.argwhere(flipmask == True)  # find indices of non zero elements 
    pinit = np.copy(ptrue) 
    return pinit, ind


# corrupts p when its -1/1 code
def corrupt_p(Np, pflip, ptrue):
    pinit, ind = corrupt_pmask(Np, pflip, ptrue)
    pinit[ind] = -1*pinit[ind] 
    return pinit


# corrupts p when its 0/1 code
def corrupt_p01(Np, pflip, ptrue):
    pinit, ind = corrupt_pmask(Np, pflip, ptrue)
    pinit[ind] = np.multiply(1-ptrue[ind], 1+ptrue[ind])
    return pinit


def train_hopfield(pbook, Npatts):
    return (1/Npatts)*(pbook[:,:Npatts])@(np.transpose(pbook[:,:Npatts])) 
    #return (pbook[:,:Npatts])@(np.transpose(pbook[:,:Npatts])) 

        
def train_gcpc(pbook, gbook, Npatts):
    return (1/Npatts)*(gbook[:,:Npatts])@(np.transpose(pbook[:,:Npatts]))   
    #return (gbook[:,:Npatts])@(np.transpose(pbook[:,:Npatts]))  


def hopfield(pinit, ptrue, Niter, W):
    p = pinit
    for i in range (Niter):
        p = np.sign(W@p);  
    return sum(abs(p-ptrue))/sum(abs(pinit-ptrue))


def gcpc(pinit, ptrue, Niter, Wgp, Wpg, gbook, lambdas):
    p = pinit
    for i in range(Niter):
        gin = Wgp@p;
        g = module_wise_NN(gin, gbook[:,:lambdas[2]], lambdas)
        p = np.sign(Wpg@g); 
    return sum(abs(p-ptrue))/sum(abs(pinit-ptrue));



def default_model(lambdas, Ng, Np, pflip, Niter, Npos, gbook, Npatts_lst, nruns):
    # avg error over Npatts
    err_hop = -1*np.ones((len(Npatts_lst), nruns))
    err_gcpc = -1*np.ones((len(Npatts_lst), nruns))

    # run for multiple seeds
    for r in range(nruns):
        print ("run: ", r)
        s = np.random.seed(r);    # set a seed

        Wpg = randn(Np,Ng);           # fixed random gc-to-pc weights
        g = np.zeros((Ng,1));         # gc activity
        p = np.zeros((Np,1));         # pc activity
        pbook = np.sign(Wpg@gbook)

        k=0
        for Npatts in Npatts_lst:
            W = np.zeros((Np,Np));      # plastic pc-pc weights
            Wgp = np.zeros((Ng,Np));    # plastic pc-to-gc weights

            # Learning patterns 
            W = train_hopfield(pbook, Npatts)
            Wgp = train_gcpc(pbook, gbook, Npatts)
            
            # Testing
            sum_hop = 0
            sum_gcpc = 0 
            for x in range(Npatts): 
                ptrue = pbook[:,x, None]                # true (noiseless) pc pattern
                pinit = corrupt_p(Np, pflip, ptrue)     # make corrupted pc pattern
                cleanup = hopfield(pinit, ptrue, Niter, W)                  # pc-pc autoassociative cleanup  
                sum_hop += cleanup
                cleanup = gcpc(pinit, ptrue, Niter, Wgp, Wpg, gbook, lambdas)   # pc-gc autoassociative cleanup
                sum_gcpc += cleanup
            err_hop[k, r] = sum_hop/Npatts
            err_gcpc[k, r] = sum_gcpc/Npatts
            k += 1   

    return err_hop, err_gcpc               



