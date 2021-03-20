# DBSCAN
Density-based spatial clustering of applications with noise (DBSCAN) is a data clustering algorithm proposed by Martin Ester, Hans-Peter Kriegel, Jörg Sander and Xiaowei Xu in 1996.
It is a density-based clustering non-parametric algorithm: given a set of points in some space, it groups together points that are closely packed together (points with many nearby neighbors), marking as outliers points that lie alone in low-density regions (whose nearest neighbors are too far away).

DBSCAN is one of the most common clustering algorithms and also most cited in scientific literature.

Code :
```python

def neighbor(X,pt,epsilon):
    N=[]
    for i,qt in enumerate(X):
        if np.linalg.norm(np.array(pt) - np.array(qt)) <= epsilon :
            N.append(i)
    return N

# input : --->
# X :  is a data list like [ [.5647,.78,.5995,....], [0.9747, 0.898, 0.5995,....], ...] . 
# individual row define a point in form of n dimentional vector 

# min_points : minimum number of point to make a Core point , 
# epsilon : minimum distance to form core point
# <---

# output : labels containing list of cluster label like [1,2,3,1,3,3,-1....] len(X) == len(labels)
    
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
```
