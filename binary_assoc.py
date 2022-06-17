# Input patterns to place cells are grid-structured, through a 
# random weight matrix that is fixed. THe goal of this code
# is to compare the power of 1) learned internal (pc-pc) associative connections to
# autoassociatively correct learned grid-driven pc patterns, versus 2)
# learned pc-to-gc connections, taken together with the fixed gc-to-pc
# weights, to autoassociatively correct the patterns. 


# grid code: binary 0/1
# place code: binary -1/1
# Works with both numpy and non-numpy version of assoc_utils


import numpy as np
import matplotlib.pyplot as plt
from src.assoc_utils_np import *
from src.data_utils import *
from src.theory_utils import *

lambdas = [7,8]             # module period
M = 2                       # num modules
Ng = 24                     # num grid cells
Np = 350                    # num place cells
pflip = 0.20                # probability of flipping a pc state to generate errors for cleanup
Niter = 15                  # number of iterations for autoassociative dynamics
Npos = nCr(Ng,M)
gbook = kbits(Ng,M)
gbook = shuffle_gbook(gbook)
Npatts_lst = np.arange(1,Npos+1,20)  # number of patterns to train on 
nruns = 10  # number of runs you want to average the results over

err_hop, err_gcpc = default_model(lambdas, Ng, Np, pflip, Niter, Npos, gbook, Npatts_lst, nruns)

# Save Data
dict = {
            "M": M,
            "lambdas": lambdas,
            "Ng": Ng,
            "Np": Np,
            "pflip": pflip,
            "Niter": Niter, 
            "Npos" : Npos,
            "gbook" : "binary",
            "Npatts_lst": Npatts_lst,
            "nruns": nruns, 
            "err_hop": err_hop,
            "err_gcpc": err_gcpc
        }

pflip = int(pflip*100)
filename="assoc_default_model"+"_gc="+str(Ng)+"_pc="+str(Np)+"_pflip="+str(pflip)+"%_runs="+str(nruns)+"_lambdas="+str(lambdas)+"_Niter"+str(Niter)
results_dir = "binary_assoc"
fname = f"{results_dir}/{filename}"
write_pkl(fname, dict)


# Analyze and Plot Data

# mean error over runs
avgerr_hop = np.mean(err_hop, axis=1)
avgerr_gcpc = np.mean(err_gcpc, axis=1)

# std dev over runs
stderr_hop = np.std(err_hop, axis=1)
stderr_gcpc = np.std(err_gcpc, axis=1)


fig1, ax1 = plt.subplots()
ax1 = plt_gcpc(ax1, Npatts_lst, avgerr_gcpc, stderr_gcpc)
savefig(fig1, ax1, f"{results_dir}/{filename}_1")
plt.show()

fig2, ax2 = plt.subplots()
ax2 = plt_hopfield(ax2, Npatts_lst, avgerr_hop, stderr_hop)
savefig(fig2, ax2, f"{results_dir}/{filename}_2")
plt.show()

fig3, ax3 = plt.subplots()
ax3 = plt_gcpc(ax3, Npatts_lst, avgerr_gcpc, stderr_gcpc)
ax3 = plt_hopfield(ax3, Npatts_lst, avgerr_hop, stderr_hop)
savefig(fig3, ax3, f"{results_dir}/{filename}_3")
plt.show()


