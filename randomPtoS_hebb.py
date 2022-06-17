import numpy as np
import matplotlib.pyplot as plt
from src.data_utils import *
from src.assoc_utils_np import *
import math
from scipy.special import erf
import sys
#np.set_printoptions(threshold=sys.maxsize)
plt.style.use('./src/presentation.mplstyle')



# # nearest neighbour
def cleanup(s, sbook):
    idx = np.argmax(sbook.T@s)
    sclean = sbook[:,idx, None]
    #sclean = sbook[:,idx]
    return sclean

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



def train_Wsp(sbook, pbook, Npatts):
    return (1/Npatts)*np.einsum('ijk, ilk -> ijl', sbook[:,:,:Npatts], pbook[:,:,:Npatts])


# computes rank of a codebook
def rank(codebook):
    return np.linalg.matrix_rank(codebook)


def nCr(n,r):
    f = math.factorial
    return f(n) // f(r) // f(n-r)


def mutual_info(m):
    a = (1+m)/2
    b = (1-m)/2
    S = - a * np.log2(a) - b * np.log2(b)
    S = np.where(m==1, np.zeros_like(S), S)
    MI = 1 - S 
    return MI


def info_hebb(N, Npatts_lst):
    p = 0.5*(1-erf(np.sqrt(N/(2*Npatts_lst))))
    MI = 1.+p*np.log2(p)+(1.-p)*np.log2(1.-p)
    return MI


nruns = 1
M = 3 
Ng = 24
Npos = nCr(Ng,M) 
Np = 300
Ns = Npos 

gbook = kbits(Ng,M)
gbook = shuffle_gbook(gbook)
Wpg = randn(nruns, Np, Ng)
pbook = np.sign(Wpg@gbook)#+

#pbook = np.sign(randn(nruns, Np, Npos))
sbook = np.sign(randn(nruns, Ns, Npos))

Npatts_lst = np.arange(1,Npos+1) #[50, 100, 200, 300, 400, 800] #np.arange(1,Npos+1)
err_sens = -1*np.ones((len(Npatts_lst), nruns))
for idx, Npatts in enumerate(Npatts_lst):
    print(idx, Npatts)
    Wsp = train_Wsp(sbook, pbook, Npatts)
    srecns = np.sign(np.einsum('rsp, rpk -> rsk', Wsp, pbook[:,:,:Npatts]))
    l1_error = np.sum(np.abs(srecns[:,:,:Npatts]-sbook[:,:,:Npatts]), axis=1)/(2*Ns)
    err_sens[idx] = np.average(l1_error, axis=1)   #avg over patterns
    #print(err_sens[idx])

# plt.figure()
# plt.imshow(Wsp[0])
# plt.colorbar()
# plt.show()
# exit()
normlizd_l1 = err_sens
m = 1 - (2 * normlizd_l1)
m = np.average(m, axis=-1) 

a = (1+m)/2
b = (1-m)/2

S = - a * np.log2(a) - b * np.log2(b)
S = np.where(m==1, np.zeros_like(S), S)
MI = 1 - S 

MIhebb_theory = info_hebb(Np, Npatts_lst)


dict = {
    "MI": MI,
    "m": m,
    "M": M,
    "err_sens": err_sens,
    "Np": Np,
    "Ng": Ng,
    "Ns": Ns,
    "Npos": Npos,
    "Npatts_lst": Npatts_lst,
    "pbook": pbook,
    "sbook": sbook
}

filename = "PtoS_hebb_Np="+str(Np)+"_Ng="+str(Ng)+"_Ns="+str(Ns)+"_runs="+str(nruns)+"_M="+str(M)+"_pbookstructured_sbookbinaryrandomnormal"
results_dir = "capacity_assoc_senstrans/dense/given_sens/RandMsparse/18C3"
fname = f"{results_dir}/{filename}"
write_pkl(fname, dict)


plt.figure(1)
#plt.plot(Npatts_lst, MI, 'r-', label="numerics")
plt.plot(Npatts_lst, MIhebb_theory, 'k--', label="theory")
plt.legend()
plt.xlabel("Number of patterns")
plt.ylabel(r"Mutual Information ($MI$)")
plt.yscale("log")
plt.xscale("log")
plt.savefig(f"continuum_results/{filename}.svg", format='svg', dpi=400, bbox_inches='tight',)
plt.savefig(f"continuum_results/{filename}.png", format='png', dpi=400, bbox_inches='tight',)
plt.show()
