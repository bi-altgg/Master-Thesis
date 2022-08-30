import random
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.linalg as la
import math,cmath
from scipy.sparse import diags
from scipy.optimize import curve_fit
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams['figure.figsize'] = [20, 8]

SMALL_SIZE = 24
MEDIUM_SIZE = 24
BIGGER_SIZE = 24
parameter_size = 10

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=parameter_size)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)

point= 8**2
colormark =[['lawngreen',"o"],['tab:cyan',"^"],['tab:olive',"o"],['tab:gray',"^"],['tab:pink',"o"],['tab:brown',"^"],['tab:purple',"o"],['tab:red',"^"],['tab:green',"o"],['tab:orange',"^"],['tab:blue',"o"]]
def func(x,a,b):

    return x**(-a)*b
sgstrn = np.sort(np.loadtxt('eigenvalue(0.0)11.dat'))
fitsite = [1]*11
for i in range(11):
    plot = np.loadtxt('(0.0)conductivity v_s strength_at_11_site'+'%1.2f'%sgstrn[i]+'.txt', dtype = 'float')
    keep_point = np.where(plot > (1.0E-18))
    plot=plot[keep_point]
    gamma_str = np.loadtxt('(0.0)x-axis11.txt',dtype = 'float')
    gamma_str=gamma_str[keep_point]
    fittingx = np.loadtxt('(0.0)x-axis11.txt',dtype = 'float')
    keep_point2 = np.where(fittingx > fitsite[i])
    fittingx = fittingx[keep_point2]
    fittingy = np.loadtxt('(0.0)conductivity v_s strength_at_11_site'+'%1.2f'%sgstrn[i]+'.txt', dtype = 'float')
    fittingy = fittingy[keep_point2]
    popt, pcov = curve_fit(func, fittingx, fittingy, maxfev=300000)
    residuals = fittingy- func(fittingx, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((fittingy - np.mean(fittingy))**2)
    print(1 - (ss_res/ss_tot))
    plt.scatter(gamma_str, plot,s = point,edgecolors = colormark[i][0],marker = colormark[i][1],c = 'w',alpha = 0.8,label =  f'$\epsilon_F = {sgstrn[i]}$')
    plt.title('Tight-Binding')
    plt.xlabel('$\gamma$')
    plt.ylabel('$G/G_o$')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
plt.savefig('conductivity_scaling_(0.0)fit11.pdf')
plt.show()

