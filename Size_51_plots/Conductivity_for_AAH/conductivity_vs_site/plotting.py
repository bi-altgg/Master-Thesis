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
plt.rcParams['figure.figsize'] = [18, 8]

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
sgstrn = [0.0,0.1,0.5,2.0]
colormark =[['green',"."],['blue',"."],['red',"."],['tab:cyan',"."],['tab:cyan',"."],['tab:olive',"."]]
ax1 = plt.subplot(131)
for i in range(4):
    plot = np.loadtxt('(0.0)datafile_for'+ str(sgstrn[i]) + '.txt', dtype = 'float')
    free_energy = np.loadtxt('freeenerg.txt',dtype = 'float')
    plt.plot(free_energy, plot,s,c = colormark[i][0],linestyle = 'dashdot',alpha = .8,label =  f'$\gamma = {sgstrn[i]}$')
    plt.title('Tight-Binding')
    plt.xlabel('Site index')
    plt.ylabel('$G/G_o$')
    plt.ylim((0.1,0.5))
    plt.yscale('log')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.text(0,0.47,'(a)')
    plt.legend(by_label.values(), by_label.keys())
    plt.grid(True)
ax2 = plt.subplot(132)
for i in range(4):
    plot = np.loadtxt('(0.5)datafile_for'+ str(sgstrn[i]) + '.txt', dtype = 'float')
    free_energy = np.loadtxt('freeenerg.txt',dtype = 'float')
    plt.plot(free_energy, plot,s,c = colormark[i][0],linestyle = 'dashdot',alpha = .8,label =  f'$\gamma = {sgstrn[i]}$')
    plt.title('AAH for 0.5')
    plt.xlabel('Site index')
    plt.yscale('log')
    plt.ylim((0.01,1.0))
    plt.text(0,0.85,'(b)')
    plt.grid(True)
ax3 = plt.subplot(133)
for i in range(4):
    plot = np.loadtxt('(1.0)datafile_for'+ str(sgstrn[i]) + '.txt', dtype = 'float')
    free_energy = np.loadtxt('freeenerg.txt',dtype = 'float')
    plt.plot(free_energy, plot,s,c = colormark[i][0],linestyle = 'dashdot',alpha = .8,)
    plt.title('AAH for 1.0')
    plt.xlabel('Site index')
    plt.yscale('log')
    plt.text(0,16,'(c)')
    plt.grid(True)
plt.savefig('Conductivity_vs_site.pdf')
plt.show()
