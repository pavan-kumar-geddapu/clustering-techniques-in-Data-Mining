import pandas as pd
from sklearn.cluster import Birch
import matplotlib.pyplot as plt
import time

data = pd.read_csv('preprocessed_data.csv', names=['pc1','pc2'])

start_time = time.time()
birch = Birch(n_clusters=2).fit(data)
time_taken = time.time() - start_time
print(time_taken)

plt.scatter(data['pc1'], data['pc2'], c= birch.labels_.astype(float), s=0.1, alpha=1)
plt.show()
