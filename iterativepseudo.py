
import numpy as np
import matplotlib.pyplot as plt
from src.assoc_utils import *
from src.data_utils import *
from src.theory_utils import *
from src.assoc_utils_np import gen_gbook
from src.sensory_utils import *


def train_pcsc(pbook, sbook, Npatts):
    return (1/Npatts)*np.einsum('ij, lj -> il', pbook[:,:Npatts], sbook[:,:Npatts])


def pseudotrain(sbook, ca1book, Npatts):
    ca1inv = np.linalg.pinv(ca1book[:,:Npatts])
    return (1/Npatts)*np.einsum('ij, jl -> il', sbook[:,:Npatts], ca1inv[:Npatts,:])  

def pseudoinverse(x):
    return np.linalg.pinv(x)

# scale matrix values to lie between 0 and 1
def normmat(W):
    normW = W - np.min(W)
    normW = normW / np.max(normW)
    return normW



m = 3
Ng = 18                   # num grid cells
Npos = nCr(Ng,m)  
Np = 200
Ns = Npos
sparsity = 1
gbook = kbits(Ng,m)
sbook = np.sign(randn(Ns, Npos))


Wpg = randn(Np, Ng)
pbook = np.sign(Wpg@gbook)   #(Np,Npos)
Wgp = train_pcsc(gbook, pbook, Npos)


# -------------------------------------------------
# WA=Y => WS = P 
# A=S, Y=P
# M=Ns, N=Np, k=Npos

M = Np #Ns
N = Ns #Np
k = Npos


epsilon = 0.01

# A = np.eye(M, M)
# Y = np.zeros((N, M))
theta = (1/epsilon**2)*np.eye(M, M)
W = np.zeros((N, M))

for i in range(k):
    ak = pbook[:,i, None]
    yk = sbook[:,i, None]
    bk = ((theta@ak) /(1+ak.T@theta@ak)).T
    theta = theta - theta@ak@bk
    W = W + (yk - W@ak)@bk



# plt.figure(1)
# plt.imshow(np.sign(W@pbook))
# plt.colorbar()


# plt.figure(2)
# plt.imshow(sbook)  
# plt.colorbar()


Wsp = pseudotrain(sbook, pbook, Npos)

plt.figure(3)
plt.imshow(np.sign(W@pbook)-np.sign(Wsp@pbook))  
plt.colorbar()

plt.figure(4)
plt.imshow(np.sign(Wsp@pbook)-sbook)  
plt.colorbar()

plt.figure(5)
plt.imshow(np.sign(W@pbook)-sbook)  
plt.colorbar()
plt.show()

exit()

#W = normmat(W)
#print(W)



# -------------------------------------------------
Wps = pseudotrain(pbook, sbook, Npos)
#Wps = normmat(Wps)
#print(Wps)

diff = W-Wps
print(diff)
plt.figure(1)
plt.imshow(W)
plt.colorbar()

plt.figure(2)
plt.imshow(Wps)
plt.colorbar()

plt.figure(3)
plt.imshow(diff)
plt.colorbar()
plt.show()
# -------------------------------------------------

