import random
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.linalg as la
import math,cmath
from scipy.sparse import diags
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
s= 15
sgstrn = [0.0,0.1,0.5,1.0,2.0,5.0,10.0]
colormark =[['green',"."],['blue',"."],['red',"."],['tab:grey',"."],['tab:cyan',"."],['tab:pink',"."],['tab:olive',"."]]
ax1 = plt.subplot(131)
for i in range(7):
    plot = np.loadtxt('(0.5)datafile_for_smal_lattice_size'+ str(sgstrn[i]) + '.txt', dtype = 'float')
    keep_point = np.where(plot > 1.0E-18)
    plot=plot[keep_point]
    free_energy = np.loadtxt('(0.5)free_energ.txt',dtype = 'float')
    free_energy=free_energy[keep_point]
    plt.scatter(free_energy, plot,s,c = colormark[i][0],marker = colormark[i][1],alpha = .8,label =  f'$\gamma = {sgstrn[i]}$')
    plt.title('($\lambda = 0.5$)')
    plt.xlabel('$\epsilon_{F}$')
    plt.ylabel('$G/G_o$')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.xlim((-3.5,+3.5))
ax2 = plt.subplot(132)
for i in range(7):
    plot = np.loadtxt('(1.0)datafile_for_smal_lattice_size'+ str(sgstrn[i]) + '.txt', dtype = 'float')
    keep_point = np.where(plot > 1.0E-18)
    plot=plot[keep_point]
    free_energy = np.loadtxt('(1.0)free_energ.txt',dtype = 'float')
    free_energy=free_energy[keep_point]
    plt.scatter(free_energy, plot,s,c = colormark[i][0],marker = colormark[i][1],alpha = .8,label =  f'$\gamma = {sgstrn[i]}$')
    plt.title('($\lambda = 1.0$)')
    plt.xlabel('$\epsilon_{F}$')
    plt.yscale('log')
    plt.grid(True)
    plt.xlim((-3.5,+3.5))
ax3 = plt.subplot(133)
for i in range(7):
    plot = np.loadtxt('(1.2)datafile_for_smal_lattice_size'+ str(sgstrn[i]) + '.txt', dtype = 'float')
    keep_point = np.where(plot > 1.0E-18)
    plot=plot[keep_point]
    free_energy = np.loadtxt('(1.2)free_energ.txt',dtype = 'float')
    free_energy=free_energy[keep_point]
    plt.scatter(free_energy, plot,s,c = colormark[i][0],marker = colormark[i][1],alpha = .8,label =  f'$\gamma = {sgstrn[i]}$')
    plt.title('($\lambda = 1.2$)')
    plt.xlabel('$\epsilon_{F}$')
    plt.yscale('log')
    plt.grid(True)
    plt.xlim((-3.5,+3.5))
plt.savefig("smal_size.pdf", dpi=300)
plt.show()
