import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pyclustering.cluster.kmedoids import kmedoids
import time

data_df = pd.read_csv('preprocessed_data.csv', names=['pc1','pc2'])
data = data_df.values
n = np.size(data, 0)
cluster_labels = np.arange(n)

initial_medoids = [1,1000]
kmedoids_instance = kmedoids(data, initial_medoids)

start_time = time.time()
kmedoids_instance.process()
time_taken = time.time() - start_time
print(time_taken)

clusters = kmedoids_instance.get_clusters()
cluster_count = 0
for cluster in clusters:
    for element in  cluster:
        cluster_labels[element] = cluster_count
    cluster_count += 1

plt.scatter(data[:,0], data[:,1], c= cluster_labels.astype(float), s=0.1, alpha=1)
plt.show()

