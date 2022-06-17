from src.data_utils import read_pkl, write_pkl
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf
plt.style.use('./src/presentation.mplstyle')


results_dir = "continuum_results/sparseconn_hopfield"
#filename = "sparsehopfield__mutualinfo_N=708_noise=0.0_iter=100_p=0.1.pkl"
#filename = "boundedhopfield__mutualinfo_N=708_noise=0.0_iter=100_bound=0.3.pkl"
filename = "sparseconnhopfield__mutualinfo_N=31623_noise=0.0_iter=100_gamma=0.0001.pkl"
fname = f"{results_dir}/{filename}" 
data = read_pkl(fname)
MI90 = data["MI"]
Npatts_lst90 = data["Npatts_list"]#/(1e-4*data["N"])

filename = "sparseconnhopfield__mutualinfo_N=10000_noise=0.0_iter=100_gamma=0.001.pkl"
fname = f"{results_dir}/{filename}" 
data = read_pkl(fname)
MI80 = data["MI"]
Npatts_lst80 = data["Npatts_list"]#/(1e-3*data["N"])

filename = "sparseconnhopfield__mutualinfo_N=3163_noise=0.0_iter=100_gamma=0.01.pkl"
fname = f"{results_dir}/{filename}" 
data = read_pkl(fname)
MI70 = data["MI"]
Npatts_lst70 = data["Npatts_list"]#/(1e-2*data["N"])


filename = "sparseconnhopfield__mutualinfo_N=317_noise=0.0_iter=100_gamma=1.pkl"
fname = f"{results_dir}/{filename}" 
data = read_pkl(fname)
MI50 = data["MI"]
Npatts_lst50 = data["Npatts_list"]#/data["N"]


filename = "MI_N = 708_input noise=0_iter=100_boundedhopfield_multiple"
plt.figure(1)
#plt.plot(Npatts_lst50, MI50, 'g-', label=r"$\gamma=1$")
plt.plot(Npatts_lst70[:20], MI70[:20], 'darkgray', label=r"$\gamma=1e-2$")
plt.plot(Npatts_lst80, MI80, 'dimgray', label=r"$\gamma=1e-3$")
plt.plot(Npatts_lst90, MI90, 'k', label=r"$\gamma=1e-4$")

plt.xlabel("Number of patterns")
plt.ylabel(r"Mutual Information ($MI$)")
plt.legend()
# plt.savefig(f"continuum_figures/figure3/sparseconnhopfield/{filename}_constantsynapses.svg", format='svg', dpi=400, bbox_inches='tight',)
# plt.savefig(f"continuum_figures/figure3/sparseconnhopfield/{filename}_constantsynapses.png", format='png', dpi=400, bbox_inches='tight',)
plt.show()