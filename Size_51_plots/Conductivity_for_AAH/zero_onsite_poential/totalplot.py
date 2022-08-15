import random
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.linalg as la
import math,cmath
from scipy.sparse import diags
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams['figure.figsize'] = [16, 10]

SMALL_SIZE = 22
MEDIUM_SIZE = 22
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)
s= 10**2
sgstrn = [0.0,0.5,1.0,5.0]
colormark =[['green',"."],['blue',"."],['red',"."],['tab:cyan',"."],['tab:cyan',"."],['tab:olive',"."]]
for i in range(4):
    plot = np.loadtxt("(0.0)datafile_for_51lattice"+ str(sgstrn[i]) + '.txt', dtype = 'float')
    keep_point = np.where(plot > 1.0E-18)
    plot=plot[keep_point]
    free_energy = np.loadtxt("(0.0)free_energ51.txt",dtype = 'float')
    free_energy=free_energy[keep_point]
    plt.scatter(free_energy, plot,s,c = colormark[i][0],marker = colormark[i][1],alpha = .8,label =  f'$\gamma = {sgstrn[i]}$')
    plt.title('Tight-Binding Model')
    plt.xlabel('$\epsilon_{F}$')
    plt.ylabel('$G/G_o$')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()    
    plt.xlim((-3.0,+3.0))
    plt.ylim((1.0e-3,2.0))
plt.savefig('Figure_of Conductivity(small_site_partial).pdf',dpi = 300)
plt.show()