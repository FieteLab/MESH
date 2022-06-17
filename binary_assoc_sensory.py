# GC-PC associative network with sensory input added
# There is associative learning between PC and sensory cells as well 


# grid code: binary 0/1
# place code: binary -1/1
# Works with both numpy and non-numpy version of assoc_utils


import numpy as np
import matplotlib.pyplot as plt
from src.assoc_utils_np import *
from src.sensory_utils import *
from src.data_utils import *
from src.theory_utils import *

lambdas = [5,6,7]           # module period
Npos = np.prod(lambdas);    # number of positions
M = 3                       # num modules
Ng = 18                     # num grid cells
Np = 300                    # num place cells
Ns = nCr(Ng,m)              # num of sensory cells
sparsity = 3                # sensory sparsity (number of 'ON' bits)
pflip = 0.0                # probability of flipping a pc state to generate errors for cleanup
Niter = 1                 # number of iterations for autoassociative dynamics
gbook = gen_gbook(lambdas, Ng, Npos)
sbook = sparse_sbook(Ns, Npos, sparsity)
Npatts_lst = np.arange(1,Npos+1,10)  # number of patterns to train on 
nruns = 20  # number of runs you want to average the results over

err_pc, err_gc, err_sens = sensory_model_gg(lambdas, Ng, Np, pflip, Niter, Npos, 
                                         gbook, Npatts_lst, nruns, Ns, sbook, sparsity)

# Save Data
dict = {
            "M": M,
            "lambdas": lambdas,
            "Ng": Ng,
            "Np": Np,
            "Ns": Ns,
            "pflip": pflip,
            "Niter": Niter, 
            "Npos" : Npos,
            "gbook" : gbook,
            "sbook" : sbook,
            "Npatts_lst": Npatts_lst,
            "nruns": nruns, 
            "err_pc": err_pc,
            "err_gc": err_gc,
            "err_sens": err_sens,
            "sparsity": sparsity
        }

pflip = int(pflip*100)
filename="assoc_sensory_model_gg"+"_gc="+str(Ng)+"_pc="+str(Np)+"_Ns="+str(Ns)+"_pflip="+str(pflip)+"%_runs="+str(nruns)+"_lambdas="+str(lambdas)+"_Niter"+str(Niter)+"_sparsity="+str(sparsity)
#results_dir = "binary_assoc_sensory/given_pc/sparse"
results_dir = "binary_assoc_sensory/sparse/given_gc/[3,4,5]"
fname = f"{results_dir}/{filename}"
write_pkl(fname, dict)


# Analyze and Plot Data

# mean error over runs
avgerr_pc = np.mean(err_pc, axis=1)
avgerr_gc = np.mean(err_gc, axis=1)
avgerr_sens = np.mean(err_sens, axis=1)

# std dev over runs
stderr_pc = np.std(err_pc, axis=1)
stderr_gc = np.std(err_gc, axis=1)
stderr_sens = np.std(err_sens, axis=1)


fig1, ax1 = plt.subplots()
ax1 = plt_gcpcsens(ax1, Npatts_lst, avgerr_pc, stderr_pc, avgerr_gc, stderr_gc, avgerr_sens, stderr_sens)
savefig(fig1, ax1, f"{results_dir}/{filename}")
plt.show()


