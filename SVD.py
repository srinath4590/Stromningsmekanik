# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 18:20:32 2026

@author: mksri
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

df=pd.read_csv('HW1Q4.csv',header= None)
matrix= df.to_numpy()

#Performing Economy SVD
U, S, VT= np.linalg.svd(matrix,full_matrices=False)

#%%
#plotting singular values (Sigma matrix)
plt.figure()
plt.plot(np.arange(1,len(S)+1),S,linewidth=2.0)
plt.xlabel('Index i')
plt.ylabel('Singular Values $\Sigma_i$')
plt.title('Singular Values of A')
plt.grid(True)
plt.gca().xaxis.set_major_locator(MultipleLocator(20))
plt.show()

#%%
#plotting first 3 Left singular vectors
k_plot=min(3,U.shape[1])
plt.figure()
for i in range (k_plot):
    plt.plot(U[:,i],label=f'$u_{{{i+1}}}$',linewidth=2.0)
plt.xlabel('Index i')
plt.ylabel('Left Singular vectors $U_i$')   
plt.title('First 3 Left Singular vectors')
plt.legend()
plt.grid(True)
plt.gca().xaxis.set_major_locator(MultipleLocator(20))
plt.show()

#%%
#plotting first 3 Right singular values
plt.figure()
for i in range(k_plot):
    plt.plot(VT[i,:],label=f'$v_{{{i+1}}}$',linewidth=2.0)
nplt.xlabel('Index i')
plt.ylabel('Right Singular vectors $V_i$')
plt.title('First 3 Right singular vectors')
plt.legend()
plt.grid(True)
plt.gca().xaxis.set_major_locator(MultipleLocator(50))
plt.gca().yaxis.set_major_locator(MultipleLocator(0.02))
plt.show()

#%%
#calculating sum of squares of first 3 singular values
k=3
num_sum= np.sum(S[:k]**2)

#calculating the frobenius norm of data projected onto first three singular vectors
V_k=VT[:k,:].T
projection= df @ V_k
frobenius_norm= np.sum(projection**2)

print('Sum of Frobenius norm of projections',frobenius_norm)
print('Sum of squares of Singular values:',num_sum)

#%%
    



