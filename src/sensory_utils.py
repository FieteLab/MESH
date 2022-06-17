from src.assoc_utils_np import *
from numpy.random import shuffle 
from src.data_utils import *
from src.sensgrid_utils import *


# generate one-hot sensory code book
# should follow Ns >= Npos
def onehot_sbook(Ns, Npos):
    if Ns >= Npos:
        return np.eye(Ns,Npos)
    else:
        print("Ns should be greater than or equal to Npos")


# generate k-sparse sensory code book
# k = sparsity
def sparse_sbook(Ns, Npos, sparsity):
    sbook = np.zeros((Ns,Npos))
    sbook[:sparsity, :] = 1
    #shuffles at each position in-place 
    # (random shuffling across neurons) 
    for i in range(Npos):
        shuffle(sbook[:,i])     
    return sbook


# converts 0/1 code to -1/1
def dense_code(sbook):
    return 2*sbook-1


def train_sensory(pbook, sbook, Npatts):
    return (1/Npatts)*np.einsum('ij, klj -> kil', sbook[:,:Npatts], pbook[:,:,:Npatts]) 


def gcpcsens_gp(pinit, ptrue, Niter, Wgp, Wpg, gbook, lambdas, Wsp, Wps, sparsity, gtrue, strue, Np):
    p = pinit
    for i in range(Niter):
        gin = Wgp@p
        sin = Wsp@p
        s = topk_binary(sin, sparsity)
        #s = np.sign(sin)
        g = module_wise_NN(gin, gbook[:,:lambdas[-1]], lambdas)  # modular net
        p = np.sign(Wpg@g + Wps@s); 
    errpc = np.sum(np.abs(p-ptrue), axis=(1,2))/(2*Np) #np.sum(np.abs(pinit-ptrue), axis=(1,2));
    errgc = np.sum(np.abs(g-gtrue), axis=(1,2))/(2*len(lambdas));
    errsens = np.sum(np.abs(s-strue), axis=(1,2))/(2*60) #(2*sparsity);
    return errpc, errgc, errsens     


def gcpcsens_gs(ptrue, Niter, Wgp, Wpg, gbook, lambdas, Wsp, Wps, sparsity, gtrue, strue, Np):
    s = strue
    p = np.sign(Wps@s)
    gin = Wgp@p
    g = module_wise_NN(gin, gbook[:,:lambdas[-1]], lambdas)  # modular net
    p = np.sign(Wpg@g + Wps@s); 
    for i in range(Niter):
        gin = Wgp@p
        sin = Wsp@p
        s = topk_binary(sin, sparsity)
        #s = np.sign(sin)
        g = module_wise_NN(gin, gbook[:,:lambdas[-1]], lambdas)  # modular net
        p = np.sign(Wpg@g + Wps@s); 
    errpc = np.sum(np.abs(p-ptrue), axis=(1,2))/(2*Np);
    errgc = np.sum(np.abs(g-gtrue), axis=(1,2))/(2*len(lambdas));
    errsens = np.sum(np.abs(s-strue), axis=(1,2))/(2*60) #(2*sparsity);
    return errpc, errgc, errsens 


def gcpcsens_gg(ptrue, Niter, Wgp, Wpg, gbook, lambdas, Wsp, Wps, sparsity, gtrue, strue, Np):
    g = gtrue
    p = np.sign(Wpg@g)
    sin = Wsp@p
    s = topk_binary(sin, sparsity)
    #s = np.sign(sin)
    p = np.sign(Wpg@g + Wps@s); 
    for i in range(Niter):
        gin = Wgp@p
        sin = Wsp@p
        #s = topk_binary(sin, sparsity)
        s = np.sign(sin)
        g = module_wise_NN(gin, gbook[:,:lambdas[-1]], lambdas)  # modular net
        p = np.sign(Wpg@g + Wps@s);  
    errpc = np.sum(np.abs(p-ptrue), axis=(1,2))/(2*Np);
    errgc = np.sum(np.abs(g-gtrue), axis=(1,2))/(2*len(lambdas));
    errsens = np.sum(np.abs(s-strue), axis=(1,2))/(2*60) #(2*sparsity);
    return errpc, errgc, errsens 



def sensory_model_gp(lambdas, Ng, Np, pflip, Niter, Npos, gbook, Npatts_lst, nruns, Ns, sbook, sparsity):
    # avg error over Npatts
    err_pc = -1*np.ones((len(Npatts_lst), nruns))
    err_sens = -1*np.ones((len(Npatts_lst), nruns))
    err_gc = -1*np.ones((len(Npatts_lst), nruns))
    M = len(lambdas)

    alpha = 0.5
    Wpg = (1-alpha)*randn(nruns, Np, Ng) / (np.sqrt(M))                       # fixed random gc-to-pc weights
    Wps = alpha*randn(nruns, Np, Ns) / (np.sqrt(sparsity))                # fixed random sensory to pc weights
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
            errpc, errgc, errsens = gcpcsens_gp(pinit, ptrue, Niter, Wgp, Wpg, gbook, 
                                            lambdas, Wsp, Wps, sparsity, gtrue, strue, Np)   
            sum_pc += errpc
            sum_gc += errgc
            sum_sens += errsens
        err_pc[k] = sum_pc/Npatts
        err_gc[k] = sum_gc/Npatts
        err_sens[k] = sum_sens/Npatts
        k += 1   
    return err_pc, err_gc, err_sens



def sensory_model_gs(lambdas, Ng, Np, pflip, Niter, Npos, gbook, Npatts_lst, nruns, Ns, sbook, sparsity):
    # avg error over Npatts
    err_pc = -1*np.ones((len(Npatts_lst), nruns))
    err_sens = -1*np.ones((len(Npatts_lst), nruns))
    err_gc = -1*np.ones((len(Npatts_lst), nruns))
    M = len(lambdas)

    alpha = 0.5
    Wpg = (1-alpha)*randn(nruns, Np, Ng) / (np.sqrt(M));                      # fixed random gc-to-pc weights
    Wps = alpha*randn(nruns, Np, Ns) / (np.sqrt(sparsity));               # fixed random sensory to pc weights
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
            
            errpc, errgc, errsens = gcpcsens_gs(ptrue, Niter, Wgp, Wpg, gbook, lambdas, 
                                                Wsp, Wps, sparsity, gtrue, strue, Np) 
            sum_pc += errpc
            sum_gc += errgc
            sum_sens += errsens
        err_pc[k] = sum_pc/Npatts
        err_gc[k] = sum_gc/Npatts
        err_sens[k] = sum_sens/Npatts
        k += 1   
    return err_pc, err_gc, err_sens  


def sensory_model_gg(lambdas, Ng, Np, pflip, Niter, Npos, gbook, Npatts_lst, nruns, Ns, sbook, sparsity):
    # avg error over Npatts
    err_pc = -1*np.ones((len(Npatts_lst), nruns))
    err_sens = -1*np.ones((len(Npatts_lst), nruns))
    err_gc = -1*np.ones((len(Npatts_lst), nruns))
    M = len(lambdas)

    alpha = 0.5
    Wpg = (1-alpha)*randn(nruns, Np, Ng) / (np.sqrt(M));                      # fixed random gc-to-pc weights
    Wps = alpha*randn(nruns, Np, Ns) / (np.sqrt(sparsity));               # fixed random sensory to pc weights
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
            
            errpc, errgc, errsens = gcpcsens_gg(ptrue, Niter, Wgp, Wpg, gbook, lambdas, 
                                                Wsp, Wps, sparsity, gtrue, strue, Np) 
            sum_pc += errpc
            sum_gc += errgc
            sum_sens += errsens
        err_pc[k] = sum_pc/Npatts
        err_gc[k] = sum_gc/Npatts
        err_sens[k] = sum_sens/Npatts
        k += 1   
    return err_pc, err_gc, err_sens  


def capacity(sensory_model, lambdas, Ng, Np_lst, pflip, Niter, Npos, gbook, Npatts_lst, nruns, Ns, sbook, sparsity):

    err_gc = -1*np.ones((len(Np_lst), len(Npatts_lst), nruns))
    err_pc = -1*np.ones((len(Np_lst), len(Npatts_lst), nruns))
    err_sens = -1*np.ones((len(Np_lst), len(Npatts_lst), nruns))
    err_senscup = -1*np.ones((len(Np_lst), len(Npatts_lst), nruns))

    l = 0
    for Np in Np_lst:
        print("l =",l)
        err_pc[l], err_gc[l], err_sens[l], err_senscup[l] = sensory_model(lambdas, Ng, Np, pflip, Niter, Npos, 
                                                gbook, Npatts_lst, nruns, Ns, sbook, sparsity)
        l = l+1

    return err_pc, err_gc, err_sens, err_senscup  
