import numpy as np
sgstrn = [-0.00,0.01,1.87,1.92]
for i in range(4):
    yaxis = np.loadtxt('(1.0) v_s strength_at_energy'+'%1.2f'%sgstrn[i]+'.txt', dtype = 'float')
    plot = np.loadtxt('(1.0)conductivity v_s strength_at_energy'+'%1.2f'%sgstrn[i]+'.txt', dtype = 'float')
    plot = [*yaxis,*plot]
    np.savetxt('1.0 strength'+'%1.2f'%sgstrn[i]+'.txt',plot)
gamma_str = np.loadtxt('1.0AAHxaxis.txt',dtype = 'float')
gamma_str2 = np.loadtxt('smallx-axis.txt',dtype = 'float')
gamma_str = [*gamma_str2,*gamma_str]
np.savetxt('xaxis.txt',gamma_str)