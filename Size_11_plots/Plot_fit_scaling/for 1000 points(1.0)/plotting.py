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
parameter_size = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=parameter_size)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)

point= 8**2
colormark =[['green',"o"],['red',"."],['orange',"^"],['tab:grey',"v"]]
def func(x,a,b):

    return x**(-a)*b
sgstrn = [0.02,-1.46,1.53, 2.00]
eigval = ['eigval','.','eigval','.']
fitsite = [8.0,8.0,7.0,6.0]
ax1 = plt.subplot(131)
for i in range(4):
    plot = np.loadtxt('(0.0)conductivity v_s strength_at_energy'+'%1.2f'%sgstrn[i]+'.txt', dtype = 'float')
    keep_point = np.where(plot > (1.0E-18))
    plot=plot[keep_point]
    gamma_str = np.loadtxt('0.0x-axis.txt',dtype = 'float')
    gamma_str=gamma_str[keep_point]
    fittingx = np.loadtxt('0.0x-axis.txt',dtype = 'float')
    keep_point2 = np.where(fittingx > fitsite[i])
    fittingx = fittingx[keep_point2]
    fittingy = np.loadtxt('(0.0)conductivity v_s strength_at_energy'+'%1.2f'%sgstrn[i]+'.txt', dtype = 'float')
    fittingy = fittingy[keep_point2]
    popt, pcov = curve_fit(func, fittingx, fittingy, maxfev=300000)
    residuals = fittingy- func(fittingx, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((fittingy - np.mean(fittingy))**2)
    plt.plot(fittingx, func(fittingx, *popt), 'b--',label='a = %1.2f, b= %1.2f' % tuple(popt))
    print(1 - (ss_res/ss_tot))
    plt.scatter(gamma_str, plot,s = point,edgecolors = colormark[i][0],marker = colormark[i][1],c = 'w',alpha = 0.8,label =  f'$\epsilon_F = {sgstrn[i]}$  {eigval[i]}')
    plt.title('Tight-Binding')
    plt.xlabel('$\gamma$')
    plt.ylabel('$G/G_o$')
    plt.yscale('log')
    plt.xscale('log')
    plt.text(60,0.7,'(a)')
    plt.legend()
sgstrn = [0.00,0.17,1.39,2.07]
eigval = ['.','.','eigval','eigval']
fitsite = [5.0,5.0,5.0,2.0]
ax2 = plt.subplot(132)
for i in range(4):
    plot = np.loadtxt('(0.5)conductivity v_s strength_at_energy'+'%1.2f'%sgstrn[i]+'.txt', dtype = 'float')
    keep_point = np.where(plot > (1.0E-18))
    plot=plot[keep_point]
    gamma_str = np.loadtxt('0.5x-axis.txt',dtype = 'float')
    gamma_str=gamma_str[keep_point]
    fittingx = np.loadtxt('0.5x-axis.txt',dtype = 'float')
    keep_point2 = np.where(fittingx > fitsite[i])
    fittingx = fittingx[keep_point2]
    fittingy = np.loadtxt('(0.5)conductivity v_s strength_at_energy'+'%1.2f'%sgstrn[i]+'.txt', dtype = 'float')
    fittingy = fittingy[keep_point2]
    popt, pcov = curve_fit(func, fittingx, fittingy, maxfev=300000)
    residuals = fittingy- func(fittingx, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((fittingy - np.mean(fittingy))**2)
    plt.plot(fittingx, func(fittingx, *popt), 'b--',label='a = %1.2f, b= %1.4f' % tuple(popt))
    print(1 - (ss_res/ss_tot))
    plt.scatter(gamma_str, plot,s = point,c = colormark[i][0],marker = colormark[i][1],alpha = .8,label =  f'$\epsilon_F = {sgstrn[i]}$  {eigval[i]}')
    plt.title('($\lambda = 0.5$)')
    plt.xlabel('$\gamma$')
    plt.yscale('log')
    plt.xscale('log')
    plt.text(60,0.7,'(b)')
    plt.legend()
sgstrn = [-0.00,0.01,1.87,1.92]
sgstrn = [0.01,1.92,1.87,-0.00]
eigval = ['.','eigval','.','eigval']
eigval = ['eigval','eigval','.','.']
fitsite = [0.2,0.8,0.2,0.2]
ax3 = plt.subplot(133)
for i in range(4):
    plot = np.loadtxt('(1.0)conductivity v_s strength_at_energy'+'%1.2f'%sgstrn[i]+'.txt', dtype = 'float')
    keep_point = np.where(plot > (1.0E-18))
    plot=plot[keep_point]
    gamma_str = np.loadtxt('x-axis.txt',dtype = 'float')
    gamma_str=gamma_str[keep_point]
    fittingx = np.loadtxt('x-axis.txt',dtype = 'float')
    keep_point2 = np.where(fittingx > fitsite[i])
    fittingx = fittingx[keep_point2]
    fittingy = np.loadtxt('(1.0)conductivity v_s strength_at_energy'+'%1.2f'%sgstrn[i]+'.txt', dtype = 'float')
    fittingy = fittingy[keep_point2]
    popt, pcov = curve_fit(func, fittingx, fittingy, maxfev=300000)
    residuals = fittingy- func(fittingx, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((fittingy - np.mean(fittingy))**2)
    plt.plot(fittingx, func(fittingx, *popt), 'b--',label='a = %1.2f, b= %1.4f' % tuple(popt))
    print(1 - (ss_res/ss_tot))
    plt.scatter(gamma_str, plot,s = point,c = colormark[i][0],marker = colormark[i][1],alpha = .8,label =  f'$\epsilon_F = {abs(sgstrn[i])}$  {eigval[i]}')
    plt.title('($\lambda = 1.0$)')
    plt.xlabel('$\gamma$')
    plt.yscale('log')
    plt.xscale('log')
    plt.text(60,0.7,'(c)')
    plt.legend()
plt.savefig('conductivity_scaling_fit.pdf')
plt.show()

