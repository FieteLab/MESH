from src.assoc_utils_np import *
from numpy.random import shuffle 
from src.data_utils import *
from src.assoc_utils_np import *
from src.sensory_utils import *


# positive threshold
# set all non zero elements to 1
# returns a 0/1 code
def threshold(x):
    return (x>0)*1

def sparseproj_gp(pinit, ptrue, Niter, Wgp, Wpg, gbook, lambdas, Wsp, Wps, sparsity, gtrue, strue, Np):
    p = pinit
    for i in range(Niter):
        gin = Wgp@p
        sin = Wsp@p
        s = topk_binary(sin, sparsity)
        g = module_wise_NN(gin, gbook[:,:lambdas[-1]], lambdas)  # modular net
        p = threshold(Wpg@g + Wps@s); 
    errpc = np.sum(np.abs(p-ptrue), axis=(1,2))/(2*Np) #np.sum(np.abs(pinit-ptrue), axis=(1,2));
    errgc = np.sum(np.abs(g-gtrue), axis=(1,2))/(2*len(lambdas));
    errsens = np.sum(np.abs(s-strue), axis=(1,2))/(2*sparsity);
    return errpc, errgc, errsens     


def sparseproj_gs(ptrue, Niter, Wgp, Wpg, gbook, lambdas, Wsp, Wps, sparsity, gtrue, strue, Np):
    s = strue
    p = np.sign(Wps@s)
    gin = Wgp@p
    g = module_wise_NN(gin, gbook[:,:lambdas[-1]], lambdas)  # modular net
    p = threshold(Wpg@g + Wps@s); 
    for i in range(Niter):
        gin = Wgp@p
        sin = Wsp@p
        s = topk_binary(sin, sparsity)
        g = module_wise_NN(gin, gbook[:,:lambdas[-1]], lambdas)  # modular net
        p = threshold(Wpg@g + Wps@s); 
    errpc = np.sum(np.abs(p-ptrue), axis=(1,2))/(2*Np);
    errgc = np.sum(np.abs(g-gtrue), axis=(1,2))/(2*len(lambdas));
    errsens = np.sum(np.abs(s-strue), axis=(1,2))/(2*sparsity);
    return errpc, errgc, errsens 


def sparseproj_gg(ptrue, Niter, Wgp, Wpg, gbook, lambdas, Wsp, Wps, sparsity, gtrue, strue, Np):
    g = gtrue
    p = threshold(Wpg@g)
    sin = Wsp@p
    s = topk_binary(sin, sparsity)
    p = threshold(Wpg@g + Wps@s); 
    for i in range(Niter):
        gin = Wgp@p
        sin = Wsp@p
        s = topk_binary(sin, sparsity)
        g = module_wise_NN(gin, gbook[:,:lambdas[-1]], lambdas)  # modular net
        p = threshold(Wpg@g + Wps@s);  
    errpc = np.sum(np.abs(p-ptrue), axis=(1,2))/(2*Np);
    errgc = np.sum(np.abs(g-gtrue), axis=(1,2))/(2*len(lambdas));
    errsens = np.sum(np.abs(s-strue), axis=(1,2))/(2*sparsity);
    return errpc, errgc, errsens 



def sens_sparse_gp(lambdas, Ng, Np, pflip, Niter, Npos, gbook, Npatts_lst, nruns, Ns, sbook, sparsity):
    # avg error over Npatts
    err_pc = -1*np.ones((len(Npatts_lst), nruns))
    err_sens = -1*np.ones((len(Npatts_lst), nruns))
    err_gc = -1*np.ones((len(Npatts_lst), nruns))
    M = len(lambdas)

    gamma = int(0.05*Np)
    Wpg = np.zeros((nruns, Np, Ng))            # fixed random gc-to-pc weights
    for i in range(Ng):
        sample = np.random.choice(np.arange(Np), size=gamma, replace=False)
        Wpg[:,sample,i] = 1
    #Wpg = Wpg/gamma                   

    Wps = np.zeros((nruns, Np, Ns))             # fixed random sensory to pc weights
    for i in range(Ns):
        sample = np.random.choice(np.arange(Np), size=gamma, replace=False)
        Wps[:,sample,i] = 1
    #Wps = Wps/gamma  

    pbook = np.sign(np.einsum('ijk,kl->ijl', Wpg, gbook) + np.einsum('ijk,kl->ijl', Wps, sbook))  # (nruns, Np, Npos)
    
    k=0
    for Npatts in Npatts_lst:
        Wgp = np.zeros((nruns, Ng, Np));    # plastic pc-to-gc weights
        Wsp=np.zeros((nruns, Ns,Np));              # plastic pc-to-sensory weights

        # Learning patterns 
        Wgp = train_gcpc(pbook, gbook, Npatts)
        Wsp = train_sensory(pbook, sbook, Npatts)

        # Testing
        sum_pc = 0
        sum_gc = 0 
        sum_sens = 0  
        for x in range(Npatts): 
            ptrue = pbook[:,:,x,None]       # true (noiseless) pc pattern
            gtrue = gbook[:,x,None]       # true (noiseless) gc pattern
            strue = sbook[:,x,None]       # true (noiseless) sensory pattern

            pinit = corrupt_p(Np, pflip, ptrue, nruns)      # make corrupted pc pattern
            errpc, errgc, errsens = sparseproj_gp(pinit, ptrue, Niter, Wgp, Wpg, gbook, 
                                            lambdas, Wsp, Wps, sparsity, gtrue, strue, Np)   
            sum_pc += errpc
            sum_gc += errgc
            sum_sens += errsens
        err_pc[k] = sum_pc/Npatts
        err_gc[k] = sum_gc/Npatts
        err_sens[k] = sum_sens/Npatts
        k += 1   
    return err_pc, err_gc, err_sens



def sens_sparse_gs(lambdas, Ng, Np, pflip, Niter, Npos, gbook, Npatts_lst, nruns, Ns, sbook, sparsity):
    # avg error over Npatts
    err_pc = -1*np.ones((len(Npatts_lst), nruns))
    err_sens = -1*np.ones((len(Npatts_lst), nruns))
    err_gc = -1*np.ones((len(Npatts_lst), nruns))
    M = len(lambdas)

    gamma = int(0.05*Np)
    Wpg = np.zeros((nruns, Np, Ng))            # fixed random gc-to-pc weights
    for i in range(Ng):
        sample = np.random.choice(np.arange(Np), size=gamma, replace=False)
        Wpg[:,sample,i] = 1
    #Wpg = Wpg/gamma                    

    Wps = np.zeros((nruns, Np, Ns))             # fixed random sensory to pc weights
    for i in range(Ns):
        sample = np.random.choice(np.arange(Np), size=gamma, replace=False)
        Wps[:,sample,i] = 1
    #Wps = Wps/gamma  

    pbook = np.sign(np.einsum('ijk,kl->ijl', Wpg, gbook) + np.einsum('ijk,kl->ijl', Wps, sbook))  # (nruns, Np, Npos)

    k=0
    for Npatts in Npatts_lst:
        Wgp = np.zeros((nruns, Ng, Np));    # plastic pc-to-gc weights
        Wsp=np.zeros((nruns, Ns,Np));              # plastic pc-to-sensory weights

        # Learning patterns 
        Wgp = train_gcpc(pbook, gbook, Npatts)
        Wsp = train_sensory(pbook, sbook, Npatts)

        # Testing
        sum_pc = 0
        sum_gc = 0 
        sum_sens = 0  
        for x in range(Npatts): 
            ptrue = pbook[:,:,x,None]       # true (noiseless) pc pattern
            gtrue = gbook[:,x,None]       # true (noiseless) gc pattern
            strue = sbook[:,x,None]       # true (noiseless) sensory pattern
            
            errpc, errgc, errsens = sparseproj_gs(ptrue, Niter, Wgp, Wpg, gbook, lambdas, 
                                                Wsp, Wps, sparsity, gtrue, strue, Np) 
            sum_pc += errpc
            sum_gc += errgc
            sum_sens += errsens
        err_pc[k] = sum_pc/Npatts
        err_gc[k] = sum_gc/Npatts
        err_sens[k] = sum_sens/Npatts
        k += 1   
    return err_pc, err_gc, err_sens  


def sens_sparse_gg(lambdas, Ng, Np, pflip, Niter, Npos, gbook, Npatts_lst, nruns, Ns, sbook, sparsity):
    # avg error over Npatts
    err_pc = -1*np.ones((len(Npatts_lst), nruns))
    err_sens = -1*np.ones((len(Npatts_lst), nruns))
    err_gc = -1*np.ones((len(Npatts_lst), nruns))
    M = len(lambdas)

    gamma = int(0.05*Np)
    Wpg = np.zeros((nruns, Np, Ng))           # fixed random gc-to-pc weights
    for i in range(Ng):
        sample = np.random.choice(np.arange(Np), size=gamma, replace=False)
        Wpg[:,sample,i] = 1
    #Wpg = Wpg/gamma                    

    Wps = np.zeros((nruns, Np, Ns))             # fixed random sensory to pc weights
    for i in range(Ns):
        sample = np.random.choice(np.arange(Np), size=gamma, replace=False)
        Wps[:,sample,i] = 1
    #Wps = Wps/gamma  

    pbook = np.sign(np.einsum('ijk,kl->ijl', Wpg, gbook) + np.einsum('ijk,kl->ijl', Wps, sbook))  # (nruns, Np, Npos)

    plt.figure(1)
    plt.imshow(Wpg[0]) 
    plt.title("Wpg")
    plt.colorbar()

    plt.figure(2)
    plt.imshow(Wps[0]) 
    plt.title("Wps")
    plt.colorbar()
    

    corr = correlation(pbook[0])
    plt.figure(3)
    plt.imshow(corr)
    plt.title("place pattern correlation")
    plt.colorbar()

    plt.show()
    exit()

    k=0
    for Npatts in Npatts_lst:
        Wgp = np.zeros((nruns, Ng, Np));    # plastic pc-to-gc weights
        Wsp=np.zeros((nruns, Ns,Np));              # plastic pc-to-sensory weights

        # Learning patterns 
        Wgp = train_gcpc(pbook, gbook, Npatts)
        Wsp = train_sensory(pbook, sbook, Npatts)

        # Testing
        sum_pc = 0
        sum_gc = 0 
        sum_sens = 0  
        for x in range(Npatts): 
            ptrue = pbook[:,:,x,None]       # true (noiseless) pc pattern
            gtrue = gbook[:,x,None]       # true (noiseless) gc pattern
            strue = sbook[:,x,None]       # true (noiseless) sensory pattern
            
            errpc, errgc, errsens = sparseproj_gg(ptrue, Niter, Wgp, Wpg, gbook, lambdas, 
                                                Wsp, Wps, sparsity, gtrue, strue, Np) 
            sum_pc += errpc
            sum_gc += errgc
            sum_sens += errsens
        err_pc[k] = sum_pc/Npatts
        err_gc[k] = sum_gc/Npatts
        err_sens[k] = sum_sens/Npatts
        k += 1   
    return err_pc, err_gc, err_sens  
  
