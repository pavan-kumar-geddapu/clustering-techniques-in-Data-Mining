import pandas as pd
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt
from numpy import unique
import time

data = pd.read_csv('preprocessed_data.csv', names=['pc1','pc2'])

start_time = time.time()
meanshift = MeanShift().fit(data)
time_taken = time.time() - start_time
print(time_taken)

print(len(unique(meanshift.labels_)))

plt.scatter(data['pc1'], data['pc2'], c= meanshift.labels_.astype(float), s=0.1, alpha=1)
plt.show()
