import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pickle
#font = fm.FontProperties(family = 'sans-serif', fname='/System/Library/Fonts/Helvetica.ttc')
plt.style.use('./src/presentation.mplstyle')


def write_pkl(filename, dict):
    with open(f"{filename}.pkl", "wb") as f:
        pickle.dump(dict, f)        


def read_pkl(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def savefig(fig, ax, filename, grid=False):
    if grid==True:
        ax.grid()
    fig.savefig(f"{filename}.svg", format='svg', dpi=400, bbox_inches='tight',)
    fig.savefig(f"{filename}.png", format='png', dpi=400, bbox_inches='tight',)
    #fig.savefig(f"{filename}.pdf", format='pdf', dpi=400, bbox_inches='tight',)
       

def add_labels(ax, title, xlabel, ylabel):
    ax.set_title(title) 
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    return ax


def plt_gcpcsens(ax, Npatts_lst, avgerr_pc, stderr_pc, avgerr_gc, stderr_gc, avgerr_sens, stderr_sens):
    ax.errorbar(Npatts_lst, avgerr_gc, yerr=stderr_gc, fmt='go-', label='grid code cleanup')
    ax.errorbar(Npatts_lst, avgerr_pc, yerr=stderr_pc, fmt='ko-', label='place code cleanup')   
    ax.errorbar(Npatts_lst, avgerr_sens, yerr=stderr_sens, fmt='co-', label='sensory code cleanup')
    add_labels(ax, "Autoassociative Cleanup", "number of patterns", "avg. cleanup error")
    print("pc: ", avgerr_pc)
    print("gc: ", avgerr_gc)
    print("sens: ", avgerr_sens)
    return ax


def plt_hopfield(ax, Npatts_lst, avgerr_hop, stderr_hop):
    ax.errorbar(Npatts_lst, avgerr_hop, yerr=stderr_hop, fmt='ro-', label='pc to pc')
    add_labels(ax, "Autoassociative Cleanup", "number of patterns", "avg. cleanup error")
    return ax


def plt_gcpc(ax, Npatts_lst, avgerr_gcpc, stderr_gcpc):
    ax.errorbar(Npatts_lst, avgerr_gcpc, yerr=stderr_gcpc, fmt='bo-', label='pc to gc')
    add_labels(ax, "Autoassociative Cleanup", "number of patterns", "avg. cleanup error")
    return ax


def plt_ggtrue(g, gtrue):
    plt.figure(1)
    plt.plot(g[0,:], 'b', label="g cleaned up")
    plt.plot(gtrue, 'r', label="gtrue")
    plt.legend()
    plt.show()


def plt_correlation(book_corr, title="Cell correlations"):
    plt.figure()
    plt.title(title, fontsize=14)
    plt.imshow(book_corr,cmap='Spectral_r')
    plt.xlabel('x_i', fontsize=12)
    plt.ylabel('x_j', fontsize=12)
    plt.colorbar()
    plt.show()


def plt_codebook(codebook, Ng, Npos):
    plt.figure()
    plt.title('Continuous 1D grid cells', fontsize=14)
    plt.imshow(codebook,cmap='Reds',extent=[0,Npos,0,Ng],aspect=1)
    plt.xlabel('position (x)', fontsize=12)
    plt.ylabel('grid cells', fontsize=12)
    plt.colorbar()
    plt.show()
    