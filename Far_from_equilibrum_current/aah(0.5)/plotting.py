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
sgstrn = [0.1, 0.5, 0.9, 1.5, 2.0]
colormark =[['green',"o"],['blue',"."],['red',"^"],['tab:grey',"v"],['tab:cyan',"s"]]
for i in range(5):
    plot = np.loadtxt("/home/bishal/Master's thesis/Codes/Far_from_equilibrum_current/aah(0.5)/current_file" + '('+ str(sgstrn[i]) + ')' + '(0.5)' +'.txt', dtype = 'float')
    gamma_str = np.loadtxt('x_axis.txt',dtype = 'float')
    plt.scatter(gamma_str, plot,s,c = colormark[i][0],marker = colormark[i][1],alpha = .8,label =  'current for $\gamma$ = ' +  str(sgstrn[i]) )
    plt.title('Tight-Binding Model')
    plt.xlabel('$\mu_R - \mu_L$')
    plt.ylabel('$\mathbf{I}$')
    plt.grid(True)
    plt.legend()
plt.show()
