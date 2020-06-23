import pandas as pd
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from numpy import unique
import time

data = pd.read_csv('preprocessed_data.csv', names=['pc1','pc2'])

start_time = time.time()
gaussian_mixture = GaussianMixture(n_components=2).fit(data)
cluster_labels = gaussian_mixture.predict(data)
time_taken = time.time() - start_time
print(time_taken)

plt.scatter(data['pc1'], data['pc2'], c= cluster_labels, s=0.1, alpha=1)
plt.show()
