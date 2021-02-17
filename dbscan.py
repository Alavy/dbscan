import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.datasets import make_blobs

from sklearn.metrics import silhouette_samples, silhouette_score, adjusted_rand_score,normalized_mutual_info_score

X,Y = make_blobs(n_samples=1000, n_features=2,centers=4,shuffle=True)

print(X.shape)
print(Y.shape)

# define constant
UNDEFINED=-2
NOISE=-1

# Finding Neighbor Of a Point
def neighbor(X,pt,epsilon):
    N=[]
    for i,qt in enumerate(X):
        if np.linalg.norm(np.array(pt) - np.array(qt)) <= epsilon :
            N.append(i)
    return N
            
def dbscan(X,min_points=4,epsilon=3.0):
    
    labels = [UNDEFINED for i in range(len(X))]
    Clus_inc =-1
    for i,pt in enumerate(X):
        
        if labels[i] != UNDEFINED:
            continue
            
        N = neighbor(X,pt,epsilon)
        
        if len(N) < min_points:
            
            labels[i]= NOISE
            
            continue
        Clus_inc = Clus_inc + 1
        
        labels[i] = Clus_inc
        
        seedset = N
        
        if i in seedset:
            seedset.remove(i)
            
        for inx in seedset:
            
            if labels[inx] == NOISE:
                labels[inx] = Clus_inc
                
            if labels[inx] != UNDEFINED:
                continue
                
            labels[inx] = Clus_inc
            
            N = neighbor(X,X[inx],epsilon)
            
            if len(N) >= min_points:
                seedset.extend(N)
    
    return labels

labels_dbscan =dbscan(X,4,1)
print(labels_dbscan)

fig = plt.figure()
fig.set_size_inches(20, 15)

ax_act = fig.add_subplot(221)
ax_act.set_title('Actual Data')
ax_act.scatter(X[:,0],X[:,1],c=Y)

ax_dbscan = fig.add_subplot(222)
ax_dbscan.set_title('DBScan Clustering')

ax_dbscan.scatter(X[:,0],X[:,1],c=labels_dbscan)

plt.show()

print(f"Silhouette Score For DBSCAN  {silhouette_score(X,labels_dbscan)}")
print(f"ARI Score For DBSCAN {adjusted_rand_score(Y,labels_dbscan)}")
print(f"NMI Score For DBSCAN {normalized_mutual_info_score(Y,labels_dbscan)}")

