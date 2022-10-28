# 2D K-Means clustering algorithm

import numpy as np
import random
from scipy.spatial import distance

""" Define K-Means Clustering Class """
class KMeansClustering:
    def __init__(self, k, dataset):
        self.k = k
        self.dataset = dataset
        self.X = [coord[0] for coord in self.dataset]
        self.Y = [coord[1] for coord in self.dataset]
        # Use Forgy Initialization
        self.centroids = random.sample(self.dataset, k)
    
    
    def __distance(self, coord1, coord2):
        """ Calculates Euclidean Distance between two coordinates
        
        Args:
            coord1/coord2 (list): list with x and y coordinates, len(coord1) == 2
        
        Returns:
            (float): Euclidean Distance
        """
        return distance.euclidean(coord1, coord2)
    
    def train(self):
        """ Training standard K-Means clustering algorithm 
        
        Returns:
            (dict): centroids (k0...kn) with cluster coordinates (list) keys for each k cluster/centroid """
        
        k_strs = ['k' + str(i) for i in range(self.k)]
        converged = False
        count = 0
        while not converged:
            count += 1
            centroid_coords = {k_str : [] for k_str in k_strs}
            for coord in self.dataset:
                if coord not in self.centroids:
                    dists = [self.__distance(self.centroids[i], coord) for i in range(self.k)]
                    centroid_coords['k' + str(dists.index(min(dists)))].append(coord)
            
            # re-assign centroids using cluster x,y mean
            mean_xs_per_k = [np.mean([centroid_coords[k_str][i][0] for i in range(len(centroid_coords[k_str]))]) for k_str in k_strs]
            mean_ys_per_k = [np.mean([centroid_coords[k_str][i][1] for i in range(len(centroid_coords[k_str]))]) for k_str in k_strs]
            new_centroids = [[np.around(mean_xs_per_k[i], decimals=2), np.around(mean_ys_per_k[i], decimals=2)] for i in range(self.k)]
            
            # check for convergance 
            if self.centroids == new_centroids:
                converged = True
            else:
            # update centroids
                self.centroids = new_centroids
    
        return centroid_coords
        
