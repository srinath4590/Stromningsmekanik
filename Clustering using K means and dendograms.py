# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 12:51:37 2026

@author: mksri
"""

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib import rcParams
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

"""
rcParams.update({'font.size':18})
plt.rcParams['figure.figsize']=[12,12]
"""

Strength_col=0
Stretch_col=1

#global variables
K=3                                       # Number of clusters to check
N_restarts=20                             # number of random initialisations to try
MAX_ITER=100                              # Maximum number of iterations
TOL=1e-8                                  # Tolerance
Random_state=42

# Function to return Y_normalised, mean, std) so that the transform can be inverted
def normalise(Y):
    mean = Y.mean(axis=0)
    std = Y.std(axis=0)
    std[std==0]=1.0
    Y_norm=(Y-mean)/std
    return Y_norm, mean, std

# Invert normalise()- bring values back to the physical scale
def denormalise(Y_norm,mean,std):
    return Y_norm*std+mean

# Implementing the Lloyds algorithm (single algorithm from one initial guess)
def assign_clusters(Y, centroids):
    k=centroids.shape[0]                        # Assigning each row of Y to nearest centroid
    distances=np.zeros((Y.shape[0],k))
    for c in range(k):
        distances[:,c]=np.linalg.norm(Y-centroids[c,:],ord=2,axis=1)
    labels=np.argmin(distances,axis=1)
    return labels, distances

def compute_inertia(Y,labels,centroids):
    inertia=0.0
    for c in range(centroids.shape[0]):
        pts= Y[labels == c,:]
        if len(pts)>0:
            inertia+=np.sum((pts-centroids[c,:])**2)
    return inertia

def K_means(Y,k,centroids,max_iter= MAX_ITER, tol=TOL):
    n_samples,n_features = Y.shape
    labels= np.zeros(n_samples,dtype=int)
    
    for i in range(max_iter):
        labels, _ = assign_clusters(Y,centroids)
        
        new_centroids = np.zeros((k,n_features))
        for c in range(k):
            pts=Y[labels == c,:]
            if len(pts)>0:
                new_centroids[c,:]=pts.mean(axis=0)
            else:
                _,distances=assign_clusters(Y,centroids)
                farthest_idx=np.argmax(np.min(distances, axis=1))
                new_centroids[c,:]=Y[farthest_idx,:]
        
        shift= np.sum(np.linalg.norm(new_centroids-centroids, axis=1))
        centroids=new_centroids
        if shift < tol:
            break    
    labels,_=assign_clusters(Y,centroids)
    inertia=compute_inertia(Y,labels,centroids)
    return centroids,labels,inertia

# Multiple restarts to avoid getting stuck in a local minimum
def kmeans_restart(Y,k,n_restart=N_restarts,random_state=Random_state):
    rng= np.random.default_rng(random_state)
    n_samples=Y.shape[0]
    
    best_inertia = None
    best_centroids = None
    best_labels = None
    all_inertias = []                     # for showing evidence of avoiding local minima
    
    for r in range (n_restart):
        idx= rng.choice(n_samples,size=k, replace=False)
        init_centroids= Y[idx,:].copy()
        
        centroids,labels,inertia=K_means(Y,k,init_centroids)
        all_inertias.append(inertia)
        
        if best_inertia is None or inertia < best_inertia:
            best_inertia=inertia
            best_centroids=centroids
            best_labels=labels
            
    return best_centroids,best_labels,best_inertia,all_inertias

data= scipy.io.loadmat('data.mat') 
Y=data['X']

# normalising features
Y_norm,mean,std=normalise(Y)
#Running K means function
centroids_norm,labels,best_inertia,all_inertias=kmeans_restart(Y_norm,k=K,n_restart=N_restarts,random_state=Random_state)

# Bringing centroids to physical (original) scale
centroids_physical=denormalise(centroids_norm, mean, std)

#plotting strength vs stretch, color-coded by cluseter, in physical scale
plt.figure(figsize=(8,5))
colors=["tab:blue","tab:orange","tab:green"]
for c in range(K):
    mask=labels==c
    plt.scatter(Y[mask,Strength_col], Y[mask,Stretch_col], c=colors[c%len(colors)], label=f"Cluster {c}", edgecolor="k", s=60)
plt.scatter(centroids_physical[:,Strength_col],centroids_physical[:,Stretch_col],c="red",marker="X", s=200, edgecolor="k",label="Centroids")
plt.xlabel("Strength")
plt.ylabel("Stretch")
plt.title(f"K-means clustering (k={K})-Strength vs Stretch")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()    

#%%
# Applying hierarchial clustering to the above algorithm
Y2=pdist(Y_norm,metric='euclidean')                               # Normalised Y matrix used for preventing domination by feature with largest numerical frame
"""
Function to perform the clustering. 
Each function initially treated as individual clusters. Clusters with least distance between them are grouped together.
Process repeates using newly merged clusters in future distance comparisons until everything has been merged into single cluster.
"""
Z=hierarchy.linkage(Y2,method='average')                          # Average method finds average distance between all pairs of points across the two clusters
thresh=0.85*np.max(Z[:,2])                                        # Threshold distance at which individual clusters are stopped being treated as seperate

plt.figure(figsize=(8,5))
dn= hierarchy.dendrogram(Z,p=100,color_threshold=thresh)
plt.axhline(y=thresh,color='r',linestyle=':',linewidth=2,label=f'Cut at {thresh:.2f}')
plt.ylabel('Linkage Distance')
plt.xlabel('Sample Index')
plt.title('Hierarchial clustering dendogram')
plt.legend()
plt.tight_layout()
plt.show()

# actual cluster labels at chosen threshold
hier_labels=hierarchy.fcluster(Z, t=3,criterion='maxclust')
hier_labels=hier_labels-1                                    #to shift to 0-indexed, like K-means labels
print("Hierarchial cluser sizes:",np.bincount(hier_labels))
print("K-means cluster sizes:   ",np.bincount(labels))

# comparing the two clusterings numerically using contingency table
contingency=np.zeros((3,3),dtype=int)
for i in range(len(labels)):
    contingency[labels[i],hier_labels[i]]+=1

print("\nContingency table (rows= K-means cluster, cols = hierarchial cluster):")
print(contingency)

# Scatter plots for hierarchial and K-means for comparison
Yp=Y_norm*std+mean                                            # Denormalising the matrix for plotting and post processing

fig,axes=plt.subplots(1,2,figsize=(12,5),sharex=True, sharey=True)
colors=["tab:blue","tab:orange","tab:green"]

for c in range(3):
    mask=labels==c
    axes[0].scatter(Yp[mask,Strength_col],Yp[mask,Stretch_col],c=colors[c],edgecolor='k',s=60,label=f'Cluster {c}')
axes[0].set_title('K-means (K=3)')
axes[0].set_xlabel('Strength')
axes[0].set_ylabel('Stretch')
axes[0].legend()

for c in range(3):
    mask=hier_labels==c
    axes[1].scatter(Yp[mask,Strength_col],Yp[mask,Stretch_col],c=colors[c],edgecolor='k',s=60,label=f'Cluster {c}')
axes[1].set_title('Hierarchial (average linkage, K=3)')
axes[1].set_xlabel('Strength')
axes[1].set_ylabel('Stretch')
axes[1].legend()

plt.tight_layout()
plt.show()