import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering

#get data
data = pd.read_csv('data.csv')

#inspect data
print data.shape
print data.head()

#get pairs values
data = data.iloc[:, 3:5].values

#cluster
cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
cluster.fit_predict(data)
dend = shc.dendrogram(shc.linkage(data, method='ward'))

#create figure
plt.title("Customer Dendograms")
plt.figure(figsize=(10, 7))

plt.figure(figsize=(10, 7))
plt.scatter(data[:,0], data[:,1], c=cluster.labels_, cmap='rainbow')

plt.show()