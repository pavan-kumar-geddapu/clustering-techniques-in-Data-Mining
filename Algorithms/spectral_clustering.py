import pandas as pd
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
import time

data = pd.read_csv('preprocessed_data.csv', names=['pc1','pc2'])

start_time = time.time()
spectral_clustering = SpectralClustering(n_clusters=2).fit(data.values)
time_taken = time.time() - start_time
print(time_taken)

plt.scatter(data['pc1'], data['pc2'], c= spectral_clustering.labels_.astype(float), s=0.1, alpha=1)
plt.show()
