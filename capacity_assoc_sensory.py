# GC-PC associative network with sensory input added
# There is associative learning between PC and sensory cells as well 


# grid code: binary 0/1
# place code: binary -1/1
# Works with both numpy and non-numpy version of assoc_utils

import sys
import numpy as np
#np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt
from src.assoc_utils_np import *
from src.sensory_utils import *
from src.data_utils import *
from src.senstranspose_utils import *
from src.theory_utils import *
# from src.sens_pcrec_utils import *
#from src.sensgrid_utils import *
#from src.sens_sparseproj_utils import *

lambdas = [7,8,10]                   # module period
M = 3                               # num modules
Ng = 18                             # num grid cells
Npos = nCr(Ng,M)                   # number of positions
Np_lst = [300] #np.arange(25, 500, 25)     # num place cells
Ns = Npos                           # num of sensory cells
sparsity = M                        # sensory sparsity (number of 'ON' bits)
pflip = 0.0                         # probability of flipping a pc state to generate errors for cleanup
Niter = 1                           # number of iterations for autoassociative dynamics
gbook = kbits(Ng,M)
gbook = shuffle_gbook(gbook)
sbook = np.sign(randn(Ns, Npos))
Npatts_lst = np.arange(1,Npos+1)  # number of patterns to train on 
nruns = 20  # number of runs you want to average the results over


err_pc, err_gc, err_sens, err_senscup = capacity(senstrans_gs, lambdas, Ng, Np_lst, pflip, Niter, Npos, 
                                         gbook, Npatts_lst, nruns, Ns, sbook, sparsity)

# Save Data
dict = {
            "M": M,
            "lambdas": lambdas,
            "Ng": Ng,
            "Np_lst": Np_lst,
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
            "err_senscup": err_senscup,
            "sparsity": sparsity
        }

pflip = int(pflip*100)
filename="sensory_model"+"_Ng="+str(Ng)+"_Ns="+str(Ns)+"_pflip="+str(pflip)+"%_runs="+str(nruns)+"_M="+str(M)+"_Niter"+str(Niter)+"_sparsity="+str(sparsity)
#results_dir = "capacity_assoc_sensory/onehot/given_sens/5_6_7"
results_dir = "capacity_assoc_senstrans/dense/given_sens/RandMsparse/18C3"
fname = f"{results_dir}/{filename}"
write_pkl(fname, dict)


# Analyze and Plot Data
from src.capacity_utils import *
errthresh = 0.001
fig, ax = plt.subplots()

# Place patterns
avg_cap, std_cap, Np_lst, Ng, lambdas = process_data(f"{filename}.pkl", results_dir, errthresh, "err_pc")
ax.errorbar(Np_lst, avg_cap, yerr=std_cap, fmt='ro--', label='Place patterns')

#print(avg_cap, std_cap)

# Sensory patterns
avg_cap, std_cap, Np_lst, Ng, lambdas = process_data(f"{filename}.pkl", results_dir, errthresh, "err_sens")
ax.errorbar(Np_lst, avg_cap, yerr=std_cap, fmt='bo--', label='Sensory patterns')

avg_cap, std_cap, Np_lst, Ng, lambdas = process_data(f"{filename}.pkl", results_dir, errthresh, "err_senscup")
ax.errorbar(Np_lst, avg_cap, yerr=std_cap, fmt='ko--', label='Sensory patterns')

#print(avg_cap, std_cap)

# Grid patterns
avg_cap, std_cap, Np_lst, Ng, lambdas = process_data(f"{filename}.pkl", results_dir, errthresh, "err_gc")
ax.errorbar(Np_lst, avg_cap, yerr=std_cap, fmt='go--', label='Grid patterns')

#print(avg_cap, std_cap)


add_labels(ax, f"Grid cells={Ng}; Periods={lambdas}; Cleanup error={errthresh}", 
            "number of place cells", "number of patterns")
savefig(fig, ax, f"{results_dir}/{filename}_thresh={errthresh}")
plt.show()


