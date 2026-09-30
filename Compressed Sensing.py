import numpy as np
import pandas as pd
from scipy.optimize import linprog

c= pd.read_csv('hw1Q5C.dat', header=None).values
b= pd.read_csv('hw1Q5b.dat', header=None).values.flatten()

# Considering x= u-v where u,v>=0 to reformulate as a linear programming problem
m,n= c.shape                                  # m= number of weighings, n= number of coins

# Single column vector of 2n rows to represent u and v 
c_obj=np.ones(2*n)                             
# C and -C matrices stacked horzontally to represent C(x-v)=b
A_eq= np.hstack([c,-c])
# c_obj is the objective function, A_eq is the constraint, and b is the RHS
# bounds= (0,None) ensures that u and v are positive, 'HIGHS' solver is selected
res=linprog(c_obj, A_eq=A_eq, b_eq=b, bounds= (0,None), method='highs') 

# res.x[:n] corresponds to u and res.x[n:] corresponds to v
x= res.x[:n]-res.x[n:]

#print("C shape:", c.shape, "b shape:", b.shape, "x shape:", x.shape)
#print("max |b|", np.max(np.abs))
#print("max|x|", np.abs(x).max())

print("Residual ||Cx-b||:", np.linalg.norm(c@x-b))

# Finding largest entry of x by magnitude
x_max= np.abs(x).max()
# Cutoff to largest entry
tol= 1e-6*x_max
# Return indices of entries of x that are larger than tol, [0] extracts index of array
fake= np.where(np.abs(x)>tol)[0]

print("Number of fradulent coins:",len(fake))
print("Weight deviations:", x[fake])
print("Indices of fake coins:", fake)
