import numpy as np

class KMeans_(object):
    """
    Parameters
    ==========
    data : array type
    
    n_clusters : int, optional, default: 3
        Number of cluster to be used
        
    tol : float, default: 1e-4
        Relative tolerance with regards to inertia to declare convergence
        
    max_iter : int, default: 300
        Maximum number of iterations of the k-means algorithm for a single run.
    """
    def __init__(self, n_clusters=3, tol=0.001, max_iter=300, seed=None):
        self.n_clusters = n_clusters
        self.tol = tol 
        self.max_iter = max_iter
        self.seed = seed
        
    def fit(self,data):
        self.data=data
        cont = True
        iteration = 0
        np.random.seed(self.seed)
        # 1. k random points of the data set are chosen to be centroids
        self.centroids = self.data[np.random.choice(len(self.data), size=self.n_clusters, replace=False),:]
    
        # 2. distances between every data point and the centroids are calculated and stored
        all_distances_old = np.array(list(map(lambda k: np.linalg.norm(self.data-self.centroids[k],axis=1),range(self.n_clusters)))).T

        # 3. based on distance calculated, each point is assigned to the nearest cluster
        self.labels = all_distances_old.argmin(axis=1)

        self.distance = np.sum(all_distances_old[:,self.labels]) # objective function

        while (iteration<self.max_iter) & cont:
            iteration += 1
            # 4. new cluster centroid positions are updated: similar to finding a mean in the point locations
            new_centroids = np.array(list(map(lambda k: np.mean(self.data[self.labels==k,:],axis=0),range(self.n_clusters)))) # new centroids
            all_distances = np.array(list(map(lambda k: np.linalg.norm(self.data-new_centroids[k],axis=1),range(self.n_clusters)))).T # compute new distance
            new_labels = all_distances.argmin(axis=1)
            new_distance = np.sum(all_distances[:,new_labels])
            
            # 5. if the new distance is lower, the process repeats from step 2
            if abs(new_distance-self.distance)/self.distance < self.tol:
                cont = False

            self.labels = new_labels
            self.distance = new_distance
            self.centroids = new_centroids
              
    def predict(self,data): # Predict the closest cluster each sample in data belongs to
        self.data=data
        labels = np.array(list(map(lambda k: np.linalg.norm(self.data-self.centroids[k],axis=1),range(self.n_clusters)))).T.argmin(axis=1)
        return labels



class KMeans(object):
    """
    Parameters
    ==========
    data : array type
    
    n_clusters : int, optional, default: 3
        Number of cluster to be used
        
    tol : float, default: 1e-4
        Relative tolerance with regards to inertia to declare convergence
        
    max_iter : int, default: 300
        Maximum number of iterations of the k-means algorithm for a single run.
        
    n_init : int, default: 10
        Number of time the k-means algorithm will be run with different centroid seeds.
    """
    def __init__(self, n_clusters=3, tol=0.001, max_iter=300, n_init=10, seed=None):
        self.n_clusters = n_clusters
        self.tol = tol 
        self.max_iter = max_iter
        self.n_init = n_init
        self.seed = seed
        
    def fit(self,data):
        self.data=data
        labels = dict.fromkeys(range(self.n_init))
        centroids = dict.fromkeys(range(self.n_init))
        distance = dict.fromkeys(range(self.n_init))
        
        for init in range(self.n_init):
            if self.seed is None:
                seed = None
            else:
                seed = init*self.seed
            tmp = KMeans_(n_clusters=self.n_clusters, tol=self.tol, max_iter=self.max_iter,seed=seed)
            tmp.fit(self.data)
            labels[init] = tmp.labels
            centroids[init] = tmp.centroids
            distance[init] = tmp.distance
        
        # output the best one
        bestkey = min(distance, key=distance.get)
        self.labels = labels[bestkey]
        self.centroids = centroids[bestkey]
        self.distance = distance[bestkey]
        
    def predict(self,data): # Predict the closest cluster each sample in data belongs to
        self.data=data
        labels = np.array(list(map(lambda k: np.linalg.norm(self.data-self.centroids[k],axis=1),range(self.n_clusters)))).T.argmin(axis=1)
        return labels