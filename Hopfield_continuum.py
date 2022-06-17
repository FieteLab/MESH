import pylab as pl
import numpy as np
import scipy.sparse as sparse
from src.data_utils import read_pkl, write_pkl



def corrupt_p(codebook,p=0.1,booktype='-11'):
    rand_indices = np.sign(np.random.uniform(size=codebook.shape)- p )
    if booktype=='-11':
        return np.multiply(codebook,rand_indices)
    elif booktype=='01':
        return abs(codebook - 0.5*(-rand_indices+1))
    elif booktype=='cts':
        return codebook + np.random.normal(0,1,size=codebook.shape)*p
    else:
        print("codebook should be -11; 01; or cts")
        return 0

def get_weights(patterns,connectivity):
    if connectivity is 'standard':
        if learning is 'hebbian':
            W = patts @ patts.T
        elif learning is 'sparsehebbian':
            prob = sparsity #np.sum(patts)/patts.shape[0]/patts.shape[1]
            W =(1/patts.shape[0])* (patts - prob) @ (patts.T - prob)
        elif learning is 'pinv':
            W= patts @ np.linalg.pinv(patts)
        W = W - pl.diag(pl.diag(W))
    else:
        N = connectivity.shape[0]
        W = sparse.lil_matrix(connectivity.shape)
        for i in range(N):
            for j in connectivity.rows[i]:
                W[i,j] = np.dot(patterns[i],patterns[j])
        W.setdiag(0)
    return W


def entropy(inlist):
    ent = pl.zeros(len(inlist))
    for idx,x in enumerate(inlist):
        if x == 0 or x == 1:
            ent[idx] = 0
        else:
            ent[idx] = -1 * ( x*np.log2(x) + (1-x)*np.log2(1-x) )
    return ent


# N = 100000
# iterations = 100
# gamma = 1e-4
# corrupt_fraction = 0.05
# Npatts_list = pl.arange(1,10)


N = 708
iterations = 100
#gamma = 1e-4
corrupt_fraction = 0.0
#Npatts_list = pl.arange(1,300,5)
#Npatts_list = pl.arange(1,5000,100)
Npatts_list = pl.arange(1,816,1)


# connectivity = sparse.lil_matrix((N,N))
# for i in range(N):
#   connectivity[i,np.random.randint(0,N,int(gamma*N))] = 1

# set connectivity to 'standard' for usual fully connected hopfield network
connectivity='standard'
# learning can be 'hebbian' or 'pinv', or 'sparsehebbian' for sparse hopfield network
learning='sparsehebbian'

# sparse hopfiled 0/1 code
#patterns = (pl.sign(pl.normal(0,1,(N,Npatts_list.max()))) + 1)/2
sparsity = 0.2
patterns = 1*(np.random.rand(N,Npatts_list.max()) > (1-sparsity))


#patterns = pl.sign(pl.normal(0,1,(N,Npatts_list.max())))
corrupt_patts = corrupt_p(patterns,p=corrupt_fraction,booktype='01')
init_overlap = pl.zeros(Npatts_list.shape[0])
final_overlap = pl.zeros(Npatts_list.shape[0])
MI_hc = pl.zeros(Npatts_list.shape[0])

for idx,Npatts in enumerate(Npatts_list):
    print(Npatts)
    patts = patterns[:,:Npatts]
    cor_patts = corrupt_patts[:,:Npatts]
    W = get_weights(patts,connectivity)
    #rep = pl.sign(W@cor_patts)

    # sparse hopfield
    theta = np.sum(W-np.diag(W), axis=1)
    theta=0.05 #0.04 #0
    # pl.hist((W@cor_patts).flatten(),bins=100)
    # pl.show()
    # exit()

    rep = (pl.sign(W@cor_patts - theta)+1)/2  
    
    #repcopy = 2*rep-1
    #pattscopy = 2*patts-1
    init_overlap[idx] = pl.average(pl.einsum('ij,ij->j',rep,patts)/N) 

    rep1 = pl.copy(rep)
    for ite in range(iterations-1):
        rep = (pl.sign(W@rep - theta)+1)/2
        if pl.sum(abs(rep - rep1))>0:
            rep1 = pl.copy(rep)
        else:
            print("converged at "+str(ite))
            break
    #repcopy = 2*rep-1
    #err = pl.average(pl.einsum('ij,ij->j',rep,patts)/N)
    err = pl.einsum('ij,ij->j',rep,patts)/N
    overlap = pl.average(err) 
    final_overlap[idx] = overlap #err
    
    q = np.sum(np.abs(rep), axis=0) / N  # sparse hopfield
    m = err
    p = np.sum(patts, axis=0)/patts.shape[0]
    P1e = 1 - (m/p)
    P0e = (q-m)/(1-p)
    MI_hc[idx] =  np.average( entropy(q) - ( p*entropy(P1e) + (1-p)*entropy(P0e) ) )


# print(init_overlap)
# print(final_overlap)

results_dir = "continuum_results/sparse_hopfield"
filename = f"sparsehopfield__mutualinfo_N={N}_noise={corrupt_fraction}_iter={iterations}_p={sparsity}"

# fig1 = pl.figure(1)
# pl.plot(Npatts_list,init_overlap, label='single, corrupt='+str(corrupt_fraction));
# pl.plot(Npatts_list,final_overlap, label='final, corrupt='+str(corrupt_fraction));
# pl.legend()
# pl.xlabel('Number of patterns')
# pl.ylabel("Overlap");
# pl.title(r"N = "+str(N)+", $W$");
# pl.show()
#fig1.savefig(f"{results_dir}/Overlap_{filename}.png")


# m = final_overlap
# a = (1+m)/2
# b = (1-m)/2

# S = - a * np.log2(a) - b * np.log2(b)
# S = np.where(m==1, np.zeros_like(S), S)

# MI_hc = 1 - S


fig2 = pl.figure(1)
pl.plot(Npatts_list,MI_hc, label='final, corrupt='+str(corrupt_fraction));
pl.legend()
pl.xlabel('Number of patterns')
pl.ylabel("MI");
pl.title(r"N = "+str(N)+", $W$");
pl.show()
fig2.savefig(f"{results_dir}/MI_{filename}.png")


data = {
    "init_overlap": init_overlap,
    "m": final_overlap,
    "MI": MI_hc,
    "Npatts_list": Npatts_list,
    "noise": corrupt_fraction,
    #"q": q
    #"gamma": gamma
}

write_pkl(f"{results_dir}/{filename}", data)
