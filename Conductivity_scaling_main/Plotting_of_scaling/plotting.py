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
colormark =[['green',"o"],['blue',"."],['red',"^"],['tab:grey',"v"]]
ax0 = plt.subplot(141)
sgstrn = [0.02,1.53,-1.46,2.00]
for i in range(4):
    plot = np.loadtxt('(0.0)conductivity v_s strength_at_energy'+'%1.2f'%sgstrn[i]+'.txt', dtype = 'float')
    keep_point = np.where(plot > 1.0E-18)
    plot=plot[keep_point]
    gamma_str = np.loadtxt('x-axis.txt',dtype = 'float')
    gamma_str=gamma_str[keep_point]
    plt.scatter(gamma_str, plot,s,c = colormark[i][0],marker = colormark[i][1],alpha = .8,label =  f'$\epsilon_F = {sgstrn[i]}$')
    plt.title('Tight-Binding')
    plt.xlabel('$\gamma$')
    plt.ylabel('$G/G_o$')
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True)
    plt.legend()
ax1 = plt.subplot(142)
sgstrn = [0.00,0.17,1.39,2.07]
for i in range(4):
    plot = np.loadtxt('(0.5)conductivity v_s strength_at_energy'+'%1.2f'%sgstrn[i]+'.txt', dtype = 'float')
    keep_point = np.where(plot > 1.0E-18)
    plot=plot[keep_point]
    gamma_str = np.loadtxt('x-axis.txt',dtype = 'float')
    gamma_str=gamma_str[keep_point]
    plt.scatter(gamma_str, plot,s,c = colormark[i][0],marker = colormark[i][1],alpha = .8,label =  f'$\epsilon_F = {sgstrn[i]}$')
    plt.title('($\lambda = 0.5$)')
    plt.xlabel('$\gamma$')
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True)
    plt.legend()
sgstrn = [0.01,1.92,1.87,-0.00]
ax2 = plt.subplot(143)
for i in range(4):
    plot = np.loadtxt('(1.0)conductivity v_s strength_at_energy'+'%1.2f'%sgstrn[i]+'.txt', dtype = 'float')
    keep_point = np.where(plot > 1.0E-18)
    plot=plot[keep_point]
    gamma_str = np.loadtxt('1.0AAHxaxis.txt',dtype = 'float')
    gamma_str=gamma_str[keep_point]
    plt.scatter(gamma_str, plot,s,c = colormark[i][0],marker = colormark[i][1],alpha = .8,label =  f'$\epsilon_F = {sgstrn[i]}$')
    plt.title('($\lambda = 1.0$)')
    plt.xlabel('$\gamma$')
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True)
    plt.legend()

sgstrn = [0.000,-0.011,-2.104,-0.143]
ax3 = plt.subplot(144)
for i in range(4):
    plot = np.loadtxt('(1.2)conductivity v_s strength_at_energy'+'%1.3f'%sgstrn[i]+'.txt', dtype = 'float')
    keep_point = np.where(plot > 1.0E-18)
    plot=plot[keep_point]
    gamma_str = np.loadtxt('x-axis.txt',dtype = 'float')
    gamma_str=gamma_str[keep_point]
    plt.scatter(gamma_str, plot,s,c = colormark[i][0],marker = colormark[i][1],alpha = .8,label =  f'$\epsilon_F = {sgstrn[i]}$')
    plt.title('($\lambda = 1.2$)')
    plt.xlabel('$\gamma$')
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True)
    plt.legend()
plt.savefig('conductivity_scaling.pdf')
plt.show()
