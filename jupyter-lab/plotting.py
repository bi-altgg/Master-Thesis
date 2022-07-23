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
colormark =[['green',"o"],['blue',"."],['red',"^"],['tab:grey',"v"],['tab:cyan',"s"]]
sgstrn = [0.1,0.5,0.9,1.5,5.0]
for i in range(5):
    plot = np.loadtxt("/home/bishal/Master's thesis/Codes/Far_from_equilibrum_current/current_file" + '('+str(sgstrn[i])+')' + '.txt', dtype = 'float')
    mu_R = np.loadtxt("/home/bishal/Master's thesis/Codes/Far_from_equilibrum_current/x_axis.txt",dtype = 'float')
    plt.scatter(mu_R, plot,s,c = colormark[i][0],marker = colormark[i][1],alpha = .8,label =  'current for' + '$\gamma$' + '=' + str(sgstrn[i]) )
    plt.title('Tight-Binding Model')
    plt.xlabel('$\mu_R - \mu_L$')
    plt.ylabel('$I$')
    plt.grid(True)
    plt.legend()
plt.savefig("current_v/s_potential.pdf")
plt.show()
