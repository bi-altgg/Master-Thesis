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
plt.rcParams['figure.figsize'] = [12, 8]

SMALL_SIZE = 18
MEDIUM_SIZE = 18
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)
s= 15
sgstrn = [0.0,0.5,1.0,5.0,10.0]
colormark =[['green',"o"],['blue',"o"],['red',"o"],['tab:cyan',"o"],['tab:pink',"o"],['tab:olive',"o"]]
for i in range(5):
    plot = np.loadtxt("/home/bishal/Master's thesis/Codes/Conductivity_for_AAH/zero_onsite_poential/(0.0)datafile_for"+ str(sgstrn[i]) + '.txt', dtype = 'float')
    keep_point = np.where(plot > 1.0E-18)
    plot=plot[keep_point]
    free_energy = np.loadtxt("/home/bishal/Master's thesis/Codes/Conductivity_for_AAH/zero_onsite_poential/(0.0)free_energ.txt",dtype = 'float')
    free_energy=free_energy[keep_point]
    plt.scatter(free_energy, plot,s,c = colormark[i][0],marker = colormark[i][1],alpha = .8,label =  f'$\gamma = {sgstrn[i]}$')
    plt.title('Tight-Binding Model')
    plt.xlabel('$\epsilon_{F}$')
    plt.ylabel('$G/G_o$')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.xlim((-2.5,+2.5))
#ax2 = plt.subplot(132)
#for i in range(7):
#    plot = np.loadtxt('(1.0)datafile_for'+ str(sgstrn[i]) + '.txt', dtype = 'float')
#    keep_point = np.where(plot > 1.0E-18)
#    plot=plot[keep_point]
#    free_energy = np.loadtxt('free_energ.txt',dtype = 'float')
#    free_energy=free_energy[keep_point]
#    plt.scatter(free_energy, plot,s,c = colormark[i][0],marker = colormark[i][1],alpha = .8,label =  f'$\gamma = {sgstrn[i]}$')
#    plt.title('($\lambda = 1.0$)')
#    plt.xlabel('$\epsilon_{F}$')
#    plt.yscale('log')
#    plt.grid(True)
#    plt.xlim((-3.5,+3.5))
#ax3 = plt.subplot(133)
#for i in range(7):
#    plot = np.loadtxt('(1.2)datafile_for'+ str(sgstrn[i]) + '.txt', dtype = 'float')
#    keep_point = np.where(plot > 1.0E-18)
#    plot=plot[keep_point]
#    free_energy = np.loadtxt('free_energ.txt',dtype = 'float')
#    free_energy=free_energy[keep_point]
#    plt.scatter(free_energy, plot,s,c = colormark[i][0],marker = colormark[i][1],alpha = .8,label =  f'$\gamma = {sgstrn[i]}$')
#    plt.title('($\lambda = 1.2$)')
#    plt.xlabel('$\epsilon_{F}$')
#    plt.yscale('log')
#    plt.grid(True)
#    plt.xlim((-3.5,+3.5))
plt.savefig('tightbindingmodel.pdf')
plt.show()
