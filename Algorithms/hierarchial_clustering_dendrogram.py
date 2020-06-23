import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc

data = pd.read_csv('preprocessed_data.csv', names=['pc1','pc2'])

plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")
dend = shc.dendrogram(shc.linkage(data, method='ward'))
plt.show()
