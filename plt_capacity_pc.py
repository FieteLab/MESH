import numpy as np
import matplotlib.pyplot as plt
from src.data_utils import read_pkl, write_pkl, savefig, add_labels
from src.capacity_utils import *
plt.style.use('./src/presentation.mplstyle')


# ----------------------------------------------------------------------------------------------------
# Randmsparse NCK pretrained on Grid patterns
fig, ax = plt.subplots()
errthresh = 0.03 
data_dir = "memscaffold_results"
results_dir = "continuum_figures/figure8" 

data_dir345 = f"{data_dir}/Ng=12/" 
filename = "assoc_capapcity_randmsparseNCK_gc=12_m=3_pflip=0.2%_runs=20_Niter1.pkl"
avg_cap345, std_cap345, Np_lst, _, lambdas = process_data(filename, data_dir345, errthresh)


data_dir567 = f"{data_dir}/Ng=18/" 
filename = "assoc_capapcity_randmsparseNCK_gc=18_m=3_pflip=0.2%_runs=20_Niter1.pkl"
avg_cap567, std_cap567, Np_lst, _, lambdas = process_data(filename, data_dir567, errthresh)


data_dir789 = f"{data_dir}/Ng=24/" 
filename = "assoc_capapcity_randmsparseNCK_gc=24_m=3_pflip=0.2%_runs=20_Niter1.pkl"
avg_cap789, std_cap789, Np_lst, _, lambdas = process_data(filename, data_dir789, errthresh)


ax.errorbar(Np_lst, avg_cap345, yerr=std_cap345, marker='o', linestyle='--', color='darkgray', label=r"$N_g$ = 12")
ax.errorbar(Np_lst, avg_cap567, yerr=std_cap567, marker='o', linestyle='--', color='dimgray', label=r"$N_g$ = 18")
ax.errorbar(Np_lst, avg_cap789, yerr=std_cap789, marker='o', linestyle='--', color='k', label=r"$N_g$ = 24")
#add_labels(ax, f"Cleanup error={errthresh}", "number of place cells", "number of patterns")
add_labels(ax, f"Cleanup error={errthresh}", "Number of hidden neurons", "Number of successfully \n recovered patterns")

savefig(fig, ax, f"{results_dir}/Capacity_three_modules_err={errthresh}_noise=20%_k=2_relaxedcap")
plt.show()
exit()

# ------------------------------------------------------------------------------
# Sensory
errthresh = 0.0
data_dir789 = "MESH_results/18C3"
filename = "sensory_model_Ng=18_Ns=816_pflip=5%_runs=20_M=3_Niter1_sparsity=3.pkl"

fig, ax = plt.subplots()

avg_cap789, std_cap789, Np_lst, _, lambdas = process_data(filename, data_dir789, errthresh, "err_gc")
ax.errorbar(Np_lst, avg_cap789, yerr=std_cap789, marker='o', linestyle='--', color='sandybrown', label='Grid')
print("test2")

avg_cap789, std_cap789, Np_lst, _, lambdas = process_data(filename, data_dir789, errthresh, "err_pc")
ax.errorbar(Np_lst, avg_cap789, yerr=std_cap789, marker='o', linestyle='--', color='mediumpurple', label='Place')
print("test1")


avg_cap789, std_cap789, Np_lst, _, lambdas = process_data(filename, data_dir789, errthresh, "err_sens")
ax.errorbar(Np_lst, avg_cap789, yerr=std_cap789, marker='o', linestyle='--', color='darkseagreen', label='Sensory')
print("test3")

avg_cap789, std_cap789, Np_lst, _, lambdas = process_data(filename, data_dir789, errthresh, "err_senscup")
ax.errorbar(Np_lst, avg_cap789, yerr=std_cap789, marker='o', linestyle='--', color='black', label='Sensory cleaned-up')
print("test4", Np_lst)

results_dir = "continuum_results" 
add_labels(ax, f"", "Number of place cells", "Capacity")
#savefig(fig, ax, f"{results_dir}/{filename}_cleanup error={errthresh}")
plt.show()
exit()

# ------------------------------------------------------------------------------ 


