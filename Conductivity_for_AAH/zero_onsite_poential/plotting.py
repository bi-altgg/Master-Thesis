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
plt.rcParams['figure.figsize'] = [18, 6]

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
sgstrn = [0.0,0.1,2.0,5.0]
colormark =[['green',"."],['blue',"."],['red',"."],['tab:cyan',"."],['tab:cyan',"."],['tab:olive',"."]]
ax1 = plt.subplot(141)
plot = np.loadtxt("/home/bishal/Master's thesis/Codes/Conductivity_for_AAH/zero_onsite_poential/(0.0)datafile_for"+ str(0.0) + '.txt', dtype = 'float')
keep_point = np.where(plot > 1.0E-18)
plot=plot[keep_point]
free_energy = np.loadtxt('(0.0)free_energ.txt',dtype = 'float')
free_energy=free_energy[keep_point]
plt.scatter(free_energy, plot,s,c = colormark[0][0],marker = colormark[0][1],alpha = 0.5)
plt.title('(a)$\gamma$ = 0.0')
plt.xlabel('$\epsilon_{F}$')
plt.ylabel('$G/G_o$')
plt.yscale('log')
plt.grid(True)
plt.xlim((-2.2,+2.2))
plt.ylim((1.0e-4,2.0))
ax2 = plt.subplot(142)
plot = np.loadtxt("/home/bishal/Master's thesis/Codes/Conductivity_for_AAH/zero_onsite_poential/(0.0)datafile_for"+ str(0.5) + '.txt', dtype = 'float')
keep_point = np.where(plot > 1.0E-18)
plot=plot[keep_point]
free_energy = np.loadtxt('(0.0)free_energ.txt',dtype = 'float')
free_energy=free_energy[keep_point]
plt.scatter(free_energy, plot,s,c = colormark[1][0],marker = colormark[1][1],alpha = 0.5)
plt.title('(a)$\gamma$ = 0.5')
plt.xlabel('$\epsilon_{F}$')
plt.ylabel('$G/G_o$')
plt.yscale('log')
plt.grid(True)
plt.xlim((-2.2,+2.2))
plt.ylim((1.0e-4,1.5))
ax3 = plt.subplot(143)
plot = np.loadtxt("/home/bishal/Master's thesis/Codes/Conductivity_for_AAH/zero_onsite_poential/(0.0)datafile_for"+ str(1.0) + '.txt', dtype = 'float')
keep_point = np.where(plot > 1.0E-18)
plot=plot[keep_point]
free_energy = np.loadtxt('(0.0)free_energ.txt',dtype = 'float')
free_energy=free_energy[keep_point]
plt.scatter(free_energy, plot,s,c = colormark[2][0],marker = colormark[2][1],alpha = 0.5)
plt.title('(a)$\gamma$ = 1.0')
plt.xlabel('$\epsilon_{F}$')
plt.ylabel('$G/G_o$')
plt.yscale('log')
plt.grid(True)
plt.xlim((-2.2,+2.2))
plt.ylim((1.0e-4,1.5))
ax4 = plt.subplot(144)
plot = np.loadtxt("/home/bishal/Master's thesis/Codes/Conductivity_for_AAH/zero_onsite_poential/(0.0)datafile_for"+ str(5.0) + '.txt', dtype = 'float')
keep_point = np.where(plot > 1.0E-18)
plot=plot[keep_point]
free_energy = np.loadtxt('(0.0)free_energ.txt',dtype = 'float')
free_energy=free_energy[keep_point]
plt.scatter(free_energy, plot,s,c = colormark[3][0],marker = colormark[3][1],alpha = 0.5)
plt.title('(a)$\gamma$ = 5.0')
plt.xlabel('$\epsilon_{F}$')
plt.ylabel('$G/G_o$')
plt.yscale('log')
plt.grid(True)
plt.xlim((-2.2,+2.2))
plt.ylim((1.0e-4,1.5))
plt.savefig('Figure_of Conductivity.pdf')
plt.show()
