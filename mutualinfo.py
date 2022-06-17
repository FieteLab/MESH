
from src.data_utils import read_pkl, write_pkl
import matplotlib.pyplot as plt
import numpy as np
#import math
from scipy.special import erf
plt.style.use('./src/presentation.mplstyle')




def info_hebb(N, Npatts_lst):
    p = 0.5*(1-erf(np.sqrt(N/(2*Npatts_lst))))
    MI = 1.+p*np.log2(p)+(1.-p)*np.log2(1.-p)
    return MI

def mutual_info(m):
    a = (1+m)/2
    b = (1-m)/2
    S = - a * np.log2(a) - b * np.log2(b)
    S = np.where(m==1, np.zeros_like(S), S)
    MI = (1 - S) 
    return MI

def mutual_info_cont(m):
    # ca1norm = 1
    # Snorm = 1
    #r = corr = np.sqrt(m) 
    r = m
    MI = np.log2(np.sqrt( 1/(1-(r**2)) ))
    #print("mutual info: ", MI)
    return MI   



results_dir = "MESH_results/18C3" 
filename = "sensory_model_Ng=18_Ns=816_pflip=5%_runs=20_M=3_Niter1_sparsity=3.pkl"


# fname = f"{results_dir}/{filename}" 
# data = read_pkl(fname)
# MI_hebb = data["MI"]
#Npatts_lsthebb = data["Npatts_lst"]
#MIhebb = info_hebb(300, Npatts_lsthebb)

# filename = "PtoS_hebb_Np=300_Ng=24_Ns=2024_runs=20_M=3_pbookstructured_sbookbinaryrandomnormal.pkl"
# fname = f"{results_dir}/{filename}" 
# data = read_pkl(fname)
# MI_struc = data["MI"]
# Npatts_lststruc = data["Npatts_lst"]

fname = f"{results_dir}/{filename}" 
data = read_pkl(fname)
err_sens = data['err_sens']
err_pc = data['err_pc']
err_senscup = data['err_senscup']
Np_lst = data["Np_lst"]
Npatts_lst = data["Npatts_lst"]
Ns = data["Ns"]
Ng = data["Ng"]
Npos = data["Npos"]
nruns = data["nruns"]
lambdas = data["lambdas"]

#MI_hebb = info_hebb(300, Npatts_lst)
# clean_err = np.mean(err_senscup, axis=-1)
# plt.figure()
# plt.plot([400,500,600, 700, 816], [400,500, 600, 700, 816], 'k-.')
# plt.plot([400, 500, 600, 700, 816], [474, 592, 690, 816, 816], 'ko-')
# plt.title("[400,500,600,700,816],[474,592,690,816,816]")
# plt.ylabel("Features with the \n correct vernoi cell")
# plt.xlabel("Number of feature neurons")
# #filename = "features_correct_vernoicell.svg"
# #plt.savefig(f"continuum_figures/figure8/{filename}", format='svg', dpi=400, bbox_inches='tight',)
# plt.show()
# print(np.where(clean_err[0]>0))
# exit()

# overlap= data["overlap_sens_original"]
# overlap_norm= data["overlap_sens_normalized"]
#overlap = np.squeeze(np.average(overlap/np.max(overlap[0,:]), axis=2))
# overlap = np.squeeze(np.average(overlap/Ns, axis=2))

# # avg over runs
#overlap = np.squeeze(np.average(overlap/np.max(overlap[0,50:]), axis=2))
# overlap_norm = np.squeeze(np.average(overlap_norm, axis=2))



# # print(len(Npatts_lst))
# print(overlap_norm.shape)
# # # plt.figure()
# # # plt.plot(Npatts_lst, overlap, "b")
# # # plt.plot(Npatts_lst, 300/Npatts_lst, "r--")
# # # #plt.ylim(0.5,1.3)
# # # plt.show()
# # # exit()
# # # print(Np_lst)
# # # exit()
# MI_randcont = mutual_info_cont(overlap_norm)
# #MI_randcont = np.where(np.isinf(MI_randcont) , 1, MI_randcont)
# # print("Test start: ", len(MI_randcont))
# # print(MI_randcont[299])
# # print(MI_randcont[300])
# # print(MI_randcont[301])

# print(len(Npatts_lst))
# print(err_sens.shape)
print(Np_lst)
# print("Ns=", Ns)
# print("Npos=", Npos)
# #print(err_sens[:5])
# MIhebb = info_hebb(Np_lst[0], Npatts_lst)



# ---------
# read autoencoder fashion mnist stuff
fname = "fashionmnist_encoded63000_mean_subtracted_norm_overlap_Np=300_Ns=500_Ng=18.pkl"
data = read_pkl(fname)
m_auto_fash = data["final_clean_overlap_list_F"]
Npatts_lst_auto_fash = data["Npatts_list"]
# ------------------------------------------------------------------------------------------------------------------------
# Mutual Information

# pick number of place cells
npidx = 0
print(Np_lst[npidx])
normlizd_l1 = err_sens[npidx]
#print(normlizd_l1[:50])

Npatts = np.array(nruns*[Npatts_lst])   # Npatts_lst repeated nruns times
Npatts = Npatts.T
print(Npatts.shape)

m = 1 - (2*normlizd_l1) # 1 - (2*normlizd_l1)
m = np.average(m, axis=-1)       #avg over 100 runs
print(m.shape)

a = (1+m)/2
b = (1-m)/2

S = - a * np.log2(a) - b * np.log2(b)
S = np.where(m==1, np.zeros_like(S), S)

MI = 1 - S 
print(MI.shape)


alpha = Npatts_lst/Ns
i = alpha*MI

Np = Np_lst[npidx]
alpha_new = (Npatts_lst*Ns) / ((2*Ns*Np)+(2*Np*Ng) )
i_new = alpha_new*MI


# data = {

#     "MI": MI,
#     "Npatts_lst": Npatts_lst,
#     "alpha": alpha,   
#     "i": i,
#     "m": m

# }

# write_pkl("18C3_Ns=500", data)
# exit()

data = read_pkl("18C3.pkl")
MI_18 = data["MI"]
Npatts_lst18 = data["Npatts_lst"]

data = read_pkl("24C3.pkl")
MI_24 = data["MI"]
Npatts_lst24 = data["Npatts_lst"]


#-------------------------
# npidx = 7
# print("TEST")
# print(Np_lst[npidx])

# normlizd_l1 = err_sens[npidx]
# #print(normlizd_l1[:50])

# m = 1 - (2*normlizd_l1) # 1 - (2 * normlizd_l1)
# m = np.average(m, axis=-1)       #avg over 100 runs
# print(m.shape)

# a = (1+m)/2
# b = (1-m)/2

# S = - a * np.log2(a) - b * np.log2(b)
# S = np.where(m==1, np.zeros_like(S), S)

# MI200 = 1 - S 
# print(MI.shape)

# #-------------------------
# npidx = 3
# print("TEST")
# print(Np_lst[npidx])

# normlizd_l1 = err_sens[npidx]
# #print(normlizd_l1[:50])

# m = 1 - (2*normlizd_l1) # 1 - (2 * normlizd_l1)
# m = np.average(m, axis=-1)       #avg over 100 runs
# print(m.shape)

# a = (1+m)/2
# b = (1-m)/2

# S = - a * np.log2(a) - b * np.log2(b)
# S = np.where(m==1, np.zeros_like(S), S)

# MI100 = 1 - S 
# print(MI.shape)

# #-------------------------
# #-------------------------
# npidx = 11
# print("TEST")
# print(Np_lst[npidx])

# normlizd_l1 = err_sens[npidx]
# #print(normlizd_l1[:50])

# m = 1 - (2*normlizd_l1) # 1 - (2 * normlizd_l1)
# m = np.average(m, axis=-1)       #avg over 100 runs
# print(m.shape)

# a = (1+m)/2
# b = (1-m)/2

# S = - a * np.log2(a) - b * np.log2(b)
# S = np.where(m==1, np.zeros_like(S), S)

# MI300 = 1 - S 
# print(MI.shape)

# #-------------------------
# #-------------------------
# npidx = -1
# print("TEST")
# print(Np_lst[npidx])

# normlizd_l1 = err_sens[npidx]
# #print(normlizd_l1[:50])

# m = 1 - (2*normlizd_l1) # 1 - (2 * normlizd_l1)
# m = np.average(m, axis=-1)       #avg over 100 runs
# print(m.shape)

# a = (1+m)/2
# b = (1-m)/2

# S = - a * np.log2(a) - b * np.log2(b)
# S = np.where(m==1, np.zeros_like(S), S)

# MI500 = 1 - S 
# print(MI.shape)
# #-------------------------
# npidx = 15
# print("TEST")
# print(Np_lst[npidx])

# normlizd_l1 = err_sens[npidx]
# #print(normlizd_l1[:50])

# m = 1 - (2*normlizd_l1) # 1 - (2 * normlizd_l1)
# m = np.average(m, axis=-1)       #avg over 100 runs
# print(m.shape)

# a = (1+m)/2
# b = (1-m)/2

# S = - a * np.log2(a) - b * np.log2(b)
# S = np.where(m==1, np.zeros_like(S), S)

# MI400 = 1 - S 
# print(MI.shape)

#-------------------------
results_dir = "continuum_results/std_hopfield"
filename = "stdhopfield__mutualinfo_N=708_noise=0.0_iter=100.pkl"
data = read_pkl(f"{results_dir}/{filename}")
MI_stdhop = data["MI"]
m_stdhop = data["m"]
Npatts_lst_stdhop = np.arange(1,816)
alpha_stdhop = Npatts_lst_stdhop/708
i_stdhop = alpha_stdhop*MI_stdhop
#-------------------------
results_dir = "continuum_results/pinv_hopfield"
filename = "pinvhopfield__mutualinfo_N=708_noise=0.05_iter=100.pkl"
data = read_pkl(f"{results_dir}/{filename}")
MI_pinvhop = data["MI"]
m_pinvhop = data["m"]
Npatts_lst_pinvhop = np.arange(1,816)
alpha_pinvhop = Npatts_lst_pinvhop/708
i_pinvhop = alpha_pinvhop*MI_pinvhop
#--------------------------
results_dir = "continuum_results/sparseconn_hopfield"
filename = "sparseconnhopfield__mutualinfo_N=3163_noise=0.0_iter=100_gamma=0.01.pkl"
data = read_pkl(f"{results_dir}/{filename}")
MI_spconhop = data["MI"]
m_spconhop = data["m"]
Npatts_lst_spco = data["Npatts_list"]
gamma = data["gamma"]
N_size = data["N"]
alpha_spco = Npatts_lst_spco/(gamma*N_size)
i_spco = alpha_spco*MI_spconhop
#--------------------------
results_dir = "continuum_results/autoencoder"
filename = "final_clean_overlap_list_Np_300_Ns_816_Ng_18.npy"
#filename="final_noisy_overlap_list_Np_300_Ns_816_Ng_18.npy"
m_auto = np.load(f"{results_dir}/{filename}")
a = (1+m_auto)/2
b = (1-m_auto)/2
S = - a * np.log2(a) - b * np.log2(b)
MI_auto = 1 - S 
Npatts_lst_auto = np.arange(50,800,50)
alpha_auto = ( (Npatts_lst_auto*Ns) ) / ( (2*Ns*Np)+(2*Np*Ng) )
#alpha_auto = Npatts_lst_auto/816
i_auto = alpha_auto*MI_auto
#--------------------------
results_dir = "continuum_results/sparse_hopfield"
filename = "sparsehopfield__mutualinfo_N=708_noise=0.0_iter=100_p=0.1.pkl"
data = read_pkl(f"{results_dir}/{filename}")
MI_spashop = data["MI"]
m_spashop = data["m"]
Npatts_lst_spashop = np.arange(1,816)
alpha_spashop = (-0.1*np.log2(0.1) - (1-0.1)*np.log2(1-0.1))*Npatts_lst_spashop/708
i_spashop = Npatts_lst_spashop*MI_spashop/708
#-------------------------
results_dir = "continuum_results/bounded_hopfield"
filename = "boundedhopfield__mutualinfo_N=708_noise=0.0_iter=100_bound=0.3.pkl"
data = read_pkl(f"{results_dir}/{filename}")
MI_bndhop = data["MI"]
m_bndhop = data["m"]
Npatts_lst_bndhop = np.arange(1,817)
alpha_bndhop = Npatts_lst_bndhop/708
i_bndhop = alpha_bndhop*MI_bndhop
#-------------------------


n = 0
plt.figure(1)
plt.plot(Npatts_lst, m[n:], linestyle='-', color='k', label="18C3, Np=300")
plt.xlabel("Number of patterns")
plt.ylabel(r"Overlap ($m$)")
# #plt.title("32C3, Np=200")
plt.legend()
#filename = "Overlap_Ng=18_Ns=816_pflip=0%_runs=20_M=3_noPseudoInverse.svg"
#plt.savefig(f"continuum_figures/figure8/{filename}", format='svg', dpi=400, bbox_inches='tight',)
#plt.show()
#exit()


plt.figure(2)
plt.plot(Npatts_lst, MI, linestyle='-', color='gray', label="18C3, Np=300")
# plt.plot(Npatts_lsthebb, MIhebb, linestyle='--', color='k', label="24C3, Np=300 theory")
# plt.plot(Npatts_lststruc, MI_struc, linestyle='-', color='rosybrown', label="{24C3, Np=300, struc P, hebb}")
plt.plot(Npatts_lst24, MI_24, linestyle='-', color='tab:red', label=r"$N_L \: = \:24$")
plt.plot(Npatts_lst18, MI_18, linestyle='-', color='tab:blue', label=r"$N_L \: = \:18$")
#plt.plot(Npatts_lst, MI, linestyle='-', color='tab:blue', label=r"$N_L \: = \: 32$")
#plt.plot(Npatts_lst24, MI_24, linestyle='-', color='tab:red', label=r"$N_L \:= \:24$")

#plt.plot(Npatts_lst, 200/Npatts_lst, linestyle='-', color='tab:gray', label=r"$2N_p/N_{patts}$")
#plt.plot(Npatts_lst18C3[300:], mutual_info_cont(np.sqrt(300/Npatts_lst18C3[300:])), linestyle='--', color='tab:gray', label=r"$S' = SP^{+}P$ cont random theory")

#plt.plot(Npatts_lst, MI_randcont, linestyle='-', color='tab:red', label= "Random cont" )
#plt.plot(Npatts_lst[299:], mutual_info_cont(np.sqrt(300/Npatts_lst[299:])), linestyle='--', color='k', label=r"$S' = SP^{+}P$ cont random theory")

#plt.plot(Npatts_lst, (2*Ns*Np+2*Np*Ng)/(Ns*Npatts_lst), linestyle='--', color='tab:gray', label=r"Upper bound: ($2N_sN_p+2N_pN_g$)/($N_s N_{patts}$)")
# plt.plot(Npatts_lst, MIhebb[n:], linestyle='-', color='k', label="{18C3, Np=300, theory}")

#plt.plot(Npatts_lst, MI100[n:], linestyle='-', color='tab:blue', label="{24C2, Np=100}")
# plt.plot(Npatts_lst, 300/Npatts_lst, linestyle='-.', color='tab:gray')
# plt.plot(Npatts_lst, MI150[n:], linestyle='-', color='k', label="{24C2, Np=150}")
# plt.plot(Npatts_lst, 150/Npatts_lst, linestyle='-.', color='tab:gray')
# plt.plot(Npatts_lst18, MI18_150[n:], linestyle='-', color='tab:orange', label="{18C3, Np=150}")
# plt.plot(Npatts_lst18, 150/Npatts_lst18, linestyle='-.', color='tab:gray')
#plt.plot(Npatts_lst18, MI18_100[n:], linestyle='-', color='tab:red', label="{18C3, Np=100}")
#plt.plot(Npatts_lst18, 100/Npatts_lst18, linestyle='-.', color='tab:gray')
#plt.plot(Npatts_lst, MI200[n:], linestyle='-', color='tab:green', label="{18C3, Np=200}")
#plt.plot(Npatts_lst, 200/Npatts_lst, linestyle='--', color='tab:green')
#plt.plot(Npatts_lst, MI300[n:], linestyle='-', color='tab:red', label="{18C3, Np=300}")
#plt.plot(Npatts_lst, 300/Npatts_lst, linestyle='--', color='tab:red')
#plt.plot(Npatts_lst, MI400[n:], linestyle='-', color='tab:purple', label="{18C3, Np=400}")
#plt.plot(Npatts_lst, 400/Npatts_lst, linestyle='--', color='tab:purple')
#plt.plot(Npatts_lst, MI500[n:], linestyle='-', color='tab:orange', label="{18C3, Np=500}")
#plt.plot(Npatts_lst, 500/Npatts_lst, linestyle='--', color='tab:orange')


# plt.plot(alpha_auto, MI_auto, linestyle='-', color='tab:orange', label=r"autoencoder") #, marker='o')
# plt.plot(alpha_stdhop, MI_stdhop, linestyle='-', color='tab:blue', label=r"std_hopfield")
# plt.plot(alpha_pinvhop, MI_pinvhop, linestyle='-', color='k', label=r"pinv_hopfield")
# plt.plot(alpha_spco, MI_spconhop, linestyle='-', color='tab:purple', label=r"sparsconn_hopfield")
# plt.plot(alpha_spashop, MI_spashop, linestyle='-', color='tab:red', label=r"sparse_hopfield")
# plt.plot(alpha_bndhop, MI_bndhop, linestyle='-', color='tab:grey', label=r"bounded_hopfield")
# plt.plot(alpha_new, MI[n:], linestyle='-', color='tab:green', label= r"$S' = sign(SP^{+}P)$")

plt.xlabel("Number of patterns")
#plt.xlabel(r"$MI_{total}\;$ stored per synapse")
#plt.ylabel(r"Mutual Information ($MI$)")
plt.ylabel(r"$MI_{perinbit}$", fontsize=23)
#filename = "MI_Np = " + str(Np_lst[npidx]) + "; lambdas = " + str(lambdas)+ "; input noise = 0"
plt.yscale("log")
plt.xscale("log")
#plt.xlim(0,1.4)
#plt.ylim(0.1, 1.2)
# plt.xlim(150,850)
plt.ylim(0.06, 1.15)

#plt.savefig(f"paper1_results/Figure4/{filename}_2.svg", format='svg', dpi=400, bbox_inches='tight',)
#plt.savefig(f"paper1_results/Figure4/{filename}_2.png", format='png', dpi=400, bbox_inches='tight',)

#plt.title("5% sensory noise")
#plt.plot(Npatts_lst24C2, MI24C2[n:], linestyle='-', color='tab:green', label="{24C2, Np=300}")
#plt.legend()
#plt.plot(alpha[n:], 1/alpha, linewidth=3, label=" 0.4 / " + r'$ \alpha $')
#plt.plot(alpha[Np_lst[npidx]],  [1], 'rd', label="No. of patterns = No. of place cells")

# plt.figure(4)
# plt.plot(alpha[n:], MI[n:], linewidth=3)
# plt.plot(alpha[Np_lst[npidx]],  [1], 'rd', label="# patterns = Np")
# plt.xlabel("fraction of patterns (" + r'$ \alpha $' + ")")
# plt.ylabel("Mutual Information (MI)")
# plt.title("Np = " + str(Np_lst[npidx]) + "; lambdas = " + str(lambdas)+ "; input noise = 0")
plt.legend()
# filename = "MI_Ng=18_24_pflip=0%_runs=20_M=3_plusnoPseudoInverse.svg"
# plt.savefig(f"continuum_figures/figure4/{filename}", format='svg', dpi=400, bbox_inches='tight',)
# plt.show()
# exit()


plt.figure(3)
#filename = "IR_Np = " + str(Np_lst[npidx]) + "; lambdas = " + str(lambdas) + "; input noise = 0"
xpoints = ypoints = plt.ylim()
plt.plot(xpoints, ypoints, 'k-.')
#plt.plot(alpha_new18C3[n:], i_new18C3[n:],  linestyle='-', color='tab:red', label=r"$N_L = 24$")
plt.plot(alpha_new[n:], i_new[n:],  linestyle='-', color='tab:green', label=r"$N_L = 18$")
plt.plot(alpha_stdhop[n:], i_stdhop[n:],  linestyle='-', color='blue', label=r"$\lambda$ = Hopfield")
plt.plot(alpha_pinvhop[n:], i_pinvhop[n:],  linestyle='-', color='k', label=r"$\lambda$ = Pseudo-inv hopfield")
plt.plot(alpha_spashop[n:], i_spashop[n:],  linestyle='-', color='tab:red', label=r"$\lambda$ = Sparse hopfield")

plt.plot(alpha_spco[n:], i_spco[n:],  linestyle='-', color='tab:purple', label=r"$\lambda$ = sparsconn hop")
plt.plot(alpha_auto[n:], i_auto[n:],  linestyle='-', color='tab:orange', label=r"$\lambda$ = Autoencoder") #, marker='o')
plt.plot(alpha_bndhop[n:], i_bndhop[n:],  linestyle='-', color='tab:gray', label=r"$\lambda$ = Bounded hopfield") 
#plt.plot(alpha789[n:], i789[n:],  linestyle='-', color='tab:blue', label=r"$\lambda$ = {18C3}")

#plt.plot(Npatts_lst, i[n:],  linestyle='-', color='tab:red', label=r"$\lambda$ = {9,10,11}")
#plt.plot(Npatts_lst789, i789[n:],  linestyle='-', color='tab:blue', label=r"$\lambda$ = {7,8,9}")
#plt.plot([Np_lst[npidx]*Ns)/Ns],  [0.5], 'kd', label="Num of patterns = Num of place cells")
#plt.xlabel("Number of patterns")
filename = "inforate_new"
plt.xlabel("Information stored per synapse (" + r'$ \alpha $' + ")")
plt.ylabel("Information recovered per \n syanpse (" + r'$ i = \alpha \times MI$' + ")")
plt.xlim(0,1.4)
plt.legend()
#plt.savefig(f"continuum_figures/figure5/inforate_all_new.svg", format='svg', dpi=400, bbox_inches='tight',)
# plt.savefig(f"continuum_figures/figure4/{filename}.png", format='png', dpi=400, bbox_inches='tight',)
plt.show()

