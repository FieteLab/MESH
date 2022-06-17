# Capacity of GC-PC associative network

# grid code: binary 0/1
# place code: binary -1/1
# only works with numpy version of assoc_utils


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
from src.assoc_utils_np import *
from src.theory_utils import *
from src.data_utils import *
plt.style.use('./src/presentation.mplstyle')


lambdas = [7,8]                   # module period
m = 2 
Ng = 24                           # num grid cells
Np_lst = np.arange(25,500,25)     # num place cells
pflip = 0.2                       # probability of flipping a pc state to generate errors for cleanup
Niter = 1                          # number of iterations for autoassociative dynamics
Npos = nCr(Ng,m)  
gbook = kbits(Ng,m)
gbook = shuffle_gbook(gbook)
nruns = 20  # number of runs you want to average the results over

err_gcpc = capacity_gcpc(lambdas, Ng, Np_lst, pflip, Niter, Npos, gbook, nruns)

# Save Data
dict = {
            "lambdas": lambdas,
            "Ng": Ng,
            "Np_lst": Np_lst,
            "pflip": pflip,
            "Niter": Niter, 
            "Npos" : Npos,
            "gbook" : "binary",
            "nruns": nruns, 
            "err_gcpc": err_gcpc,
            "network": "modular"
        }

# pflip = int(pflip*100)
filename="assoc_capapcity_randmsparseNCK"+"_gc="+str(Ng)+"_m="+str(m)+"_pflip="+str(pflip)+"%_runs="+str(nruns)+"_Niter="+str(Niter)
#filename="assoc_capacity_mod"+"_gc="+str(Ng)+"_runs="+str(nruns)+"_lambdas="+str(lambdas)+"_Niter"+str(Niter)+"_NpattsEqualNpos"
results_dir = "capacity_assoc/results/randmsparseNCK/7_8_9"
fname = f"{results_dir}/{filename}"
write_pkl(fname, dict)


# Analyze and Plot Data
errthresh = 0.03    # 3% error
capacity = -1*np.ones((len(Np_lst), nruns))

valid = err_gcpc <= errthresh   # bool
for Np in range(len(Np_lst)):
    for r in range(nruns):
        lst = np.argwhere(valid[Np,:,r] == True)
        if len(lst) == 0:
              capacity[Np,r] = 0
        else:      
              capacity[Np,r] = lst[-1]
avg_cap = np.mean(capacity, axis=1)   # mean error over runs
std_cap = stats.sem(capacity, axis=1)    # std dev over runs


fig, ax = plt.subplots()
ax.errorbar(Np_lst, avg_cap, yerr=std_cap, fmt='ko--', label='randmsparseNCK network')
add_labels(ax, f"Grid cells={Ng}; Grid periods={lambdas}; thresh={errthresh}", "number of place cells", "number of patterns")
savefig(fig, ax, f"{results_dir}/{filename}")
plt.show()



