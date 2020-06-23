import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pyclustering.cluster.clique import clique, clique_visualizer
import time

data_df = pd.read_csv('preprocessed_data.csv', names=['pc1','pc2'])
data = data_df.values
n = np.size(data, 0)
cluster_labels = np.arange(n)
for i in range(n):
    cluster_labels[i] = -1

# create CLIQUE algorithm for processing
intervals = 50  # defines amount of cells in grid in each dimension
threshold = 30   # lets consider each point as non-outlier
clique_instance = clique(data, intervals, threshold)

start_time = time.time()
clique_instance.process()
time_taken = time.time() - start_time
print(time_taken)

clusters = clique_instance.get_clusters()
cluster_count = 0
for cluster in clusters:
    for element in  cluster:
        cluster_labels[element] = cluster_count
    cluster_count += 1
print(cluster_count)

x=[]
y=[]
labels = []
for i in range(n):
    if cluster_labels[i] != -1:
        x.append(data[i][0])
        y.append(data[i][1])
        labels.append(cluster_labels[i])
print(len(x))

plt.scatter(x, y, c= labels, s=0.1, alpha=1)
plt.show()
