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
plt.rcParams['figure.figsize'] = [24, 6]

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
sgstrn = [0.1, 0.5, 0.9, 1.5, 2.0]
colormark =[['green',"o"],['blue',"."],['red',"^"],['tab:grey',"v"],['tab:cyan',"s"],['tab:purple',"s"]]
ax1 = plt.subplot(131)
for i in range(5):
    plot = np.loadtxt("current_file" + '('+ str(sgstrn[i]) + ')' + '(0.0)' +'.txt', dtype = 'float')
    print(plot)
    gamma_str = np.loadtxt("x_axis.txt",dtype = 'float')
    print(gamma_str)
    plt.scatter(gamma_str, plot,s,c = colormark[i][0],marker = colormark[i][1],alpha = .8,label =  'Current for $\gamma$ = ' +  str(sgstrn[i]) )
    plt.title('Tight-Binding Model')
    plt.xlabel('$\mu_R - \mu_L$')
    plt.ylabel('$\mathbf{I}$')
    plt.grid(True)
    plt.legend()
ax1 = plt.subplot(132)
for i in range(5):
    plot = np.loadtxt("current_file" + '('+ str(sgstrn[i]) + ')' + '(0.5)' +'.txt', dtype = 'float')
    gamma_str = np.loadtxt("x_axis.txt",dtype = 'float')
    plt.scatter(gamma_str, plot,s,c = colormark[i][0],marker = colormark[i][1],alpha = .8,label =  'Current for $\gamma$ = ' +  str(sgstrn[i]) )
    plt.title('AAH with $\lambda = 0.5$')
    plt.xlabel('$\mu_R - \mu_L$')
    plt.grid(True)
ax1 = plt.subplot(133)
for i in range(5):
    plot = np.loadtxt("current_file" + '('+ str(sgstrn[i]) + ')' + '(1.0)' +'.txt', dtype = 'float')
    gamma_str = np.loadtxt("x_axis.txt",dtype = 'float')
    plt.scatter(gamma_str, plot,s,c = colormark[i][0],marker = colormark[i][1],alpha = .8,label =  'Current for $\gamma$ = ' +  str(sgstrn[i]) )
    plt.title('AAH with $\lambda = 1.0$')
    plt.xlabel('$\mu_R - \mu_L$')
    plt.grid(True)
plt.savefig("current_v_spotential(negative-positive).pdf")
plt.show()
