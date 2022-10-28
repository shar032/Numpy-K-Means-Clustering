# Train K-means clustering on dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Kmeans import KMeansClustering

# read dataset of 2D coordinates
dataset = pd.read_csv('Kmeans_dataset.csv')

# parse data to as list of lists 
data = [[np.around(dataset['x'][i], decimals = 3), np.around(dataset['y'][i], decimals = 3)] 
    for i in range(dataset.shape[0])]

# define metrics
k = 3

# instantiate K-Means Clustering class 
kmeans = KMeansClustering(k = k, dataset = data)

# train model and store output dictionary of centroid keys and coordinates values
centroid_dict = kmeans.train()

# parse results to plot
for key in list(centroid_dict.keys()):
    vars()[key + 'x'] = [coord[0] for coord in centroid_dict[key]]
    vars()[key + 'y'] = [coord[1] for coord in centroid_dict[key]]

# plot results with cluster labels
for i in range(4):
    sns.scatterplot(vars()['k' + str(i) + 'x'], vars()['k' + str(i) + 'y'], label = 'Cluster' +  str(i))
plt.xlabel('x', fontsize = 15)
plt.ylabel('y', fontsize = 15)
plt.title('Scatter plot with Clusters', fontsize = 15)

