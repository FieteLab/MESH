from src.assoc_utils_np import *
from src.sensory_utils import *
from src.assoc_utils_np import *
from src.sensory_utils import train_sensory


def senspcrec_gs(ptrue, Niter, Wgp, Wpg, gbook, lambdas, Wsp, Wps, Wpp, sparsity, gtrue, strue, Np):
    s = strue
    p = np.sign(Wps@s)
    p = np.sign(Wpp@p)
    gin = Wgp@p
    g = module_wise_NN(gin, gbook[:,:lambdas[-1]], lambdas)  # modular net
    p = np.sign(Wpg@g + Wps@s)
    p = np.sign(Wpp@p)
    for i in range(Niter):
        gin = Wgp@p
        sin = Wsp@p
        s = topk_binary(sin, sparsity)
        g = module_wise_NN(gin, gbook[:,:lambdas[-1]], lambdas)  # modular net
        p = np.sign(Wpg@g + Wps@s); 
        p = np.sign(Wpp@p)
    errpc = np.sum(np.abs(p-ptrue), axis=(1,2))/(2*Np);
    errgc = np.sum(np.abs(g-gtrue), axis=(1,2))/(2*len(lambdas));
    errsens = np.sum(np.abs(s-strue), axis=(1,2))/(2*sparsity);
    return errpc, errgc, errsens 


def senspcrec_gg(ptrue, Niter, Wgp, Wpg, gbook, lambdas, Wsp, Wps, Wpp, sparsity, gtrue, strue, Np):
    g = gtrue
    p = np.sign(Wpg@g)
    p = np.sign(Wpp@p)
    sin = Wsp@p
    s = topk_binary(sin, sparsity)
    p = np.sign(Wpg@g + Wps@s); 
    p = np.sign(Wpp@p)
    for i in range(Niter):
        gin = Wgp@p
        sin = Wsp@p
        s = topk_binary(sin, sparsity)
        g = module_wise_NN(gin, gbook[:,:lambdas[-1]], lambdas)  # modular net
        p = np.sign(Wpg@g + Wps@s) 
        p = np.sign(Wpp@p) 
    errpc = np.sum(np.abs(p-ptrue), axis=(1,2))/(2*Np);
    errgc = np.sum(np.abs(g-gtrue), axis=(1,2))/(2*len(lambdas));
    errsens = np.sum(np.abs(s-strue), axis=(1,2))/(2*sparsity);
    return errpc, errgc, errsens 


def senspcrec_gp(pinit, ptrue, Niter, Wgp, Wpg, gbook, lambdas, Wsp, Wps, Wpp, sparsity, gtrue, strue, Np):
    p = pinit
    for i in range(Niter):
        gin = Wgp@p
        sin = Wsp@p
        s = topk_binary(sin, sparsity)
        g = module_wise_NN(gin, gbook[:,:lambdas[-1]], lambdas)  # modular net
        p = np.sign(Wpg@g + Wps@s); 
        p = np.sign(Wpp@p)   
    errpc = np.sum(np.abs(p-ptrue), axis=(1,2))/(2*Np) #np.sum(np.abs(pinit-ptrue), axis=(1,2));
    errgc = np.sum(np.abs(g-gtrue), axis=(1,2))/(2*len(lambdas));
    errsens = np.sum(np.abs(s-strue), axis=(1,2))/(2*sparsity);

    return errpc, errgc, errsens     



def sensmod_pcrec_gs(lambdas, Ng, Np, pflip, Niter, Npos, gbook, Npatts_lst, nruns, Ns, sbook, sparsity):
    # avg error over Npatts
    err_pc = -1*np.ones((len(Npatts_lst), nruns))
    err_sens = -1*np.ones((len(Npatts_lst), nruns))
    err_gc = -1*np.ones((len(Npatts_lst), nruns))
    M = len(lambdas)

    Wpg = randn(nruns, Np, Ng) / (np.sqrt(M));                      # fixed random gc-to-pc weights
    Wps = randn(nruns, Np, Ns) / (np.sqrt(sparsity));               # fixed random sensory to pc weights
                          
    
    pbook = np.sign(np.einsum('ijk,kl->ijl', Wpg, gbook) + np.einsum('ijk,kl->ijl', Wps, sbook))  # (nruns, Np, Npos)

    k=0
    for Npatts in Npatts_lst:
        Wgp = np.zeros((nruns, Ng, Np));    # plastic pc-to-gc weights
        Wsp = np.zeros((nruns, Ns,Np));     # plastic pc-to-sensory weights
        Wpp = np.zeros((nruns, Np, Np))     # plastic pc-to-pc weights

        # Learning patterns 
        Wgp = train_gcpc(pbook, gbook, Npatts)
        Wsp = train_sensory(pbook, sbook, Npatts)
        Wpp = train_hopfield(pbook, Npatts)

        # Testing
        sum_pc = 0
        sum_gc = 0 
        sum_sens = 0  
        for x in range(Npatts): 
            ptrue = pbook[:,:,x,None]       # true (noiseless) pc pattern
            gtrue = gbook[:,x,None]       # true (noiseless) gc pattern
            strue = sbook[:,x,None]       # true (noiseless) sensory pattern
            
            errpc, errgc, errsens = senspcrec_gs(ptrue, Niter, Wgp, Wpg, gbook, lambdas, 
                                                Wsp, Wps, Wpp, sparsity, gtrue, strue, Np) 
            sum_pc += errpc
            sum_gc += errgc
            sum_sens += errsens
        err_pc[k] = sum_pc/Npatts
        err_gc[k] = sum_gc/Npatts
        err_sens[k] = sum_sens/Npatts
        k += 1   
    return err_pc, err_gc, err_sens  


def sensmod_pcrec_gp(lambdas, Ng, Np, pflip, Niter, Npos, gbook, Npatts_lst, nruns, Ns, sbook, sparsity):

    # avg error over Npatts
    err_pc = -1*np.ones((len(Npatts_lst), nruns))
    err_sens = -1*np.ones((len(Npatts_lst), nruns))
    err_gc = -1*np.ones((len(Npatts_lst), nruns))
    M = len(lambdas)

    Wpg = randn(nruns, Np, Ng) / (np.sqrt(M))                       # fixed random gc-to-pc weights
    Wps = randn(nruns, Np, Ns) / (np.sqrt(sparsity))                # fixed random sensory to pc weights
    pbook = np.sign(np.einsum('ijk,kl->ijl', Wpg, gbook) + np.einsum('ijk,kl->ijl', Wps, sbook))  # (nruns, Np, Npos)

    #pbook = np.sign(randn(1,500,60))

    # print(pbook.shape)
    # corr = correlation(pbook[0])
    # plt.figure()
    # plt.imshow(corr)
    # plt.colorbar()
    # plt.show()

    k=0
    for Npatts in Npatts_lst:
        Wgp = np.zeros((nruns, Ng, Np));    # plastic pc-to-gc weights
        Wsp=np.zeros((nruns, Ns, Np));              # plastic pc-to-sensory weights
        Wpp = np.zeros((nruns, Np, Np))     # plastic pc-to-pc weights

        # Learning patterns 
        Wgp = train_gcpc(pbook, gbook, Npatts)
        Wsp = train_sensory(pbook, sbook, Npatts)
        Wpp = train_hopfield(pbook, Npatts)

        # Testing
        sum_pc = 0
        sum_gc = 0 
        sum_sens = 0  
        for x in range(Npatts): 
            ptrue = pbook[:,:,x,None]       # true (noiseless) pc pattern
            gtrue = gbook[:,x,None]       # true (noiseless) gc pattern
            strue = sbook[:,x,None]       # true (noiseless) sensory pattern

            pinit = corrupt_p(Np, pflip, ptrue, nruns)      # make corrupted pc pattern
            errpc, errgc, errsens = senspcrec_gp(pinit, ptrue, Niter, Wgp, Wpg, gbook, 
                                            lambdas, Wsp, Wps, Wpp, sparsity, gtrue, strue, Np)   
            sum_pc += errpc
            sum_gc += errgc
            sum_sens += errsens
        err_pc[k] = sum_pc/Npatts
        err_gc[k] = sum_gc/Npatts
        err_sens[k] = sum_sens/Npatts
        k += 1   
    return err_pc, err_gc, err_sens   


def sensmod_pcrec_gg(lambdas, Ng, Np, pflip, Niter, Npos, gbook, Npatts_lst, nruns, Ns, sbook, sparsity):
    # avg error over Npatts
    err_pc = -1*np.ones((len(Npatts_lst), nruns))
    err_sens = -1*np.ones((len(Npatts_lst), nruns))
    err_gc = -1*np.ones((len(Npatts_lst), nruns))
    M = len(lambdas)

    Wpg = randn(nruns, Np, Ng) / (np.sqrt(M));                      # fixed random gc-to-pc weights
    Wps = randn(nruns, Np, Ns) / (np.sqrt(sparsity));               # fixed random sensory to pc weights
    pbook = np.sign(np.einsum('ijk,kl->ijl', Wpg, gbook) + np.einsum('ijk,kl->ijl', Wps, sbook))  # (nruns, Np, Npos)

    k=0
    for Npatts in Npatts_lst:
        Wgp = np.zeros((nruns, Ng, Np));    # plastic pc-to-gc weights
        Wsp=np.zeros((nruns, Ns,Np));              # plastic pc-to-sensory weights
        Wpp = np.zeros((nruns, Np, Np))     # plastic pc-to-pc weights

        # Learning patterns 
        Wgp = train_gcpc(pbook, gbook, Npatts)
        Wsp = train_sensory(pbook, sbook, Npatts)
        Wpp = train_hopfield(pbook, Npatts)

        # Testing
        sum_pc = 0
        sum_gc = 0 
        sum_sens = 0  
        for x in range(Npatts): 
            ptrue = pbook[:,:,x,None]       # true (noiseless) pc pattern
            gtrue = gbook[:,x,None]       # true (noiseless) gc pattern
            strue = sbook[:,x,None]       # true (noiseless) sensory pattern
            
            errpc, errgc, errsens = senspcrec_gg(ptrue, Niter, Wgp, Wpg, gbook, lambdas, 
                                                Wsp, Wps, Wpp, sparsity, gtrue, strue, Np) 
            sum_pc += errpc
            sum_gc += errgc
            sum_sens += errsens
        err_pc[k] = sum_pc/Npatts
        err_gc[k] = sum_gc/Npatts
        err_sens[k] = sum_sens/Npatts
        k += 1   
    return err_pc, err_gc, err_sens      