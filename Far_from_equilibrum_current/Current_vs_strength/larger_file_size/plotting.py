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
plt.rcParams['figure.figsize'] = [15, 10]

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

plot = np.loadtxt("current_file0.15.txt", dtype = 'float')
gamma_str = np.loadtxt("x_axis.txt",dtype = 'float')
plt.scatter(gamma_str, plot,c = 'green',marker = "o",alpha = .8,label =  '$\mu_R - \mu_L = 0.15$')
plt.title('Tight-Binding Model')
plt.xlabel('$\gamma$')
plt.ylabel('$I$')
plt.xscale('log')
plt.yscale('log')
plt.grid(True)
plot = np.loadtxt("current_file0.55.txt", dtype = 'float')
gamma_str = np.loadtxt("x_axis.txt",dtype = 'float')
plt.scatter(gamma_str, plot,c = 'blue',marker = "s",alpha = .8,label =  '$\mu_R - \mu_L = 0.55$')
plt.title('Tight-Binding Model')
plt.xlabel('$\gamma$')
plt.ylabel('$I$')
plt.xscale('log')
plt.yscale('log')
plt.grid(True)
plot = np.loadtxt("current_file1.05.txt", dtype = 'float')
gamma_str = np.loadtxt("x_axis.txt",dtype = 'float')
plt.scatter(gamma_str, plot,c = 'red',marker = "^",alpha = .8,label =  '$\mu_R - \mu_L = 1.05$')
plt.title('Tight-Binding Model')
plt.xlabel('$\gamma$')
plt.ylabel('$I$')
plt.xscale('log')
plt.yscale('log')
plt.grid(True)
plt.legend()
plt.savefig('Current_conductivity_scaling.pdf')
plt.show()