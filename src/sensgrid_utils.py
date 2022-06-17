from src.assoc_utils_np import *
from numpy.random import shuffle 
from src.data_utils import *
from src.assoc_utils_np import *


def train_sensory(pbook, sbook, Npatts):
    return (1/Npatts)*np.einsum('ij, klj -> kil', sbook[:,:Npatts], pbook[:,:,:Npatts]) 


def train_gcsc(sbook, gbook, Npatts):
    return (1/Npatts)*np.einsum('ij, kj -> ik', gbook[:,:Npatts], sbook[:,:Npatts]) 


def dynamics_gs(ptrue, Niter, Wgp, Wpg, gbook, lambdas, Wsp, Wgs, sparsity, gtrue, strue, Np):
    s = strue
    gin = Wgs@s
    g = module_wise_NN(gin, gbook[:,:lambdas[-1]], lambdas)
    p = np.sign(Wpg@g)
    for i in range(Niter):
        gin = Wgp@p
        sin = Wsp@p
        s = topk_binary(sin, sparsity)
        g = module_wise_NN(gin, gbook[:,:lambdas[-1]], lambdas)  # modular net
        p = np.sign(Wpg@g)
    errpc = np.sum(np.abs(p-ptrue), axis=(1,2))/(2*Np);
    errgc = np.sum(np.abs(g-gtrue), axis=(1,2))/(2*len(lambdas));
    errsens = np.sum(np.abs(s-strue), axis=(1,2))/(2*sparsity);
    return errpc, errgc, errsens 


def dynamics_gg(ptrue, Niter, Wgp, Wpg, gbook, lambdas, Wsp, Wgs, sparsity, gtrue, strue, Np):
    g = gtrue
    p = np.sign(Wpg@g)
    for i in range(Niter):
        gin = Wgp@p
        sin = Wsp@p
        s = topk_binary(sin, sparsity)
        g = module_wise_NN(gin, gbook[:,:lambdas[-1]], lambdas)  # modular net
        p = np.sign(Wpg@g)   
    errpc = np.sum(np.abs(p-ptrue), axis=(1,2))/(2*Np);
    errgc = np.sum(np.abs(g-gtrue), axis=(1,2))/(2*len(lambdas));
    errsens = np.sum(np.abs(s-strue), axis=(1,2))/(2*sparsity);
    return errpc, errgc, errsens 


def dynamics_gp(pinit, ptrue, Niter, Wgp, Wpg, gbook, lambdas, Wsp, Wgs, sparsity, gtrue, strue, Np):
    p = pinit
    for i in range(Niter):
        gin = Wgp@p
        sin = Wsp@p
        s = topk_binary(sin, sparsity)
        g = module_wise_NN(gin, gbook[:,:lambdas[-1]], lambdas)  # modular net 
        p = np.sign(Wpg@g) 
    errpc = np.sum(np.abs(p-ptrue), axis=(1,2))/(2*Np) #np.sum(np.abs(pinit-ptrue), axis=(1,2));
    errgc = np.sum(np.abs(g-gtrue), axis=(1,2))/(2*len(lambdas));
    errsens = np.sum(np.abs(s-strue), axis=(1,2))/(2*sparsity);
    return errpc, errgc, errsens    



def sensgrid_gs(lambdas, Ng, Np, pflip, Niter, Npos, gbook, Npatts_lst, nruns, Ns, sbook, sparsity):
    # avg error over Npatts
    err_pc = -1*np.ones((len(Npatts_lst), nruns))
    err_sens = -1*np.ones((len(Npatts_lst), nruns))
    err_gc = -1*np.ones((len(Npatts_lst), nruns))
    M = len(lambdas)

    Wpg = randn(nruns, Np, Ng) / (np.sqrt(M));                      # fixed random gc-to-pc weights
    pbook = np.sign(np.einsum('ijk,kl->ijl', Wpg, gbook))  # (nruns, Np, Npos)


    k=0
    for Npatts in Npatts_lst:
        Wgp = np.zeros((nruns, Ng, Np));        # plastic pc-to-gc weights
        Wsp=np.zeros((nruns, Ns,Np));           # plastic pc-to-sensory weights

        # Learning patterns 
        Wgp = train_gcpc(pbook, gbook, Npatts)
        Wsp = train_sensory(pbook, sbook, Npatts)
        Wgs = train_gcsc(sbook, gbook, Npatts)
        Wgs = np.repeat(Wgs[np.newaxis,:,:], nruns, axis=0)

        # Testing
        sum_pc = 0
        sum_gc = 0 
        sum_sens = 0  
        for x in range(Npatts): 
            ptrue = pbook[:,:,x,None]       # true (noiseless) pc pattern
            gtrue = gbook[:,x,None]       # true (noiseless) gc pattern
            strue = sbook[:,x,None]       # true (noiseless) sensory pattern
            
            errpc, errgc, errsens = dynamics_gs(ptrue, Niter, Wgp, Wpg, gbook, lambdas, 
                                                Wsp, Wgs, sparsity, gtrue, strue, Np) 
            sum_pc += errpc
            sum_gc += errgc
            sum_sens += errsens
        err_pc[k] = sum_pc/Npatts
        err_gc[k] = sum_gc/Npatts
        err_sens[k] = sum_sens/Npatts
        k += 1   
    return err_pc, err_gc, err_sens  


def sensgrid_gg(lambdas, Ng, Np, pflip, Niter, Npos, gbook, Npatts_lst, nruns, Ns, sbook, sparsity):
    # avg error over Npatts
    err_pc = -1*np.ones((len(Npatts_lst), nruns))
    err_sens = -1*np.ones((len(Npatts_lst), nruns))
    err_gc = -1*np.ones((len(Npatts_lst), nruns))
    M = len(lambdas)

    Wpg = randn(nruns, Np, Ng) / (np.sqrt(M));                      # fixed random gc-to-pc weights
    pbook = np.sign(np.einsum('ijk,kl->ijl', Wpg, gbook))  # (nruns, Np, Npos)

    k=0
    for Npatts in Npatts_lst:
        Wgp = np.zeros((nruns, Ng, Np));    # plastic pc-to-gc weights
        Wsp=np.zeros((nruns, Ns,Np));              # plastic pc-to-sensory weights

        # Learning patterns 
        Wgp = train_gcpc(pbook, gbook, Npatts)
        Wsp = train_sensory(pbook, sbook, Npatts)
        Wgs = train_gcsc(sbook, gbook, Npatts)
        Wgs = np.repeat(Wgs[np.newaxis,:,:], nruns, axis=0)

        # Testing
        sum_pc = 0
        sum_gc = 0 
        sum_sens = 0  
        for x in range(Npatts): 
            ptrue = pbook[:,:,x,None]       # true (noiseless) pc pattern
            gtrue = gbook[:,x,None]       # true (noiseless) gc pattern
            strue = sbook[:,x,None]       # true (noiseless) sensory pattern
            
            errpc, errgc, errsens = dynamics_gg(ptrue, Niter, Wgp, Wpg, gbook, lambdas, 
                                                Wsp, Wgs, sparsity, gtrue, strue, Np) 
            sum_pc += errpc
            sum_gc += errgc
            sum_sens += errsens
        err_pc[k] = sum_pc/Npatts
        err_gc[k] = sum_gc/Npatts
        err_sens[k] = sum_sens/Npatts
        k += 1   
    return err_pc, err_gc, err_sens      


def sensgrid_gp(lambdas, Ng, Np, pflip, Niter, Npos, gbook, Npatts_lst, nruns, Ns, sbook, sparsity):
    # avg error over Npatts
    err_pc = -1*np.ones((len(Npatts_lst), nruns))
    err_sens = -1*np.ones((len(Npatts_lst), nruns))
    err_gc = -1*np.ones((len(Npatts_lst), nruns))
    M = len(lambdas)

    Wpg = randn(nruns, Np, Ng) / (np.sqrt(M))                       # fixed random gc-to-pc weights
    pbook = np.sign(np.einsum('ijk,kl->ijl', Wpg, gbook))   # (nruns, Np, Npos)

    k=0
    for Npatts in Npatts_lst:
        Wgp = np.zeros((nruns, Ng, Np));    # plastic pc-to-gc weights
        Wsp=np.zeros((nruns, Ns,Np));              # plastic pc-to-sensory weights

        # Learning patterns 
        Wgp = train_gcpc(pbook, gbook, Npatts)
        Wsp = train_sensory(pbook, sbook, Npatts)
        Wgs = train_gcsc(sbook, gbook, Npatts)
        Wgs = np.repeat(Wgs[np.newaxis,:,:], nruns, axis=0)

        # Testing
        sum_pc = 0
        sum_gc = 0 
        sum_sens = 0  
        for x in range(Npatts): 
            ptrue = pbook[:,:,x,None]       # true (noiseless) pc pattern
            gtrue = gbook[:,x,None]       # true (noiseless) gc pattern
            strue = sbook[:,x,None]       # true (noiseless) sensory pattern

            pinit = corrupt_p(Np, pflip, ptrue, nruns)      # make corrupted pc pattern
            errpc, errgc, errsens = dynamics_gp(pinit, ptrue, Niter, Wgp, Wpg, gbook, 
                                            lambdas, Wsp, Wgs, sparsity, gtrue, strue, Np)   
            sum_pc += errpc
            sum_gc += errgc
            sum_sens += errsens
        err_pc[k] = sum_pc/Npatts
        err_gc[k] = sum_gc/Npatts
        err_sens[k] = sum_sens/Npatts
        k += 1   
    return err_pc, err_gc, err_sens
