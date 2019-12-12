# Hierarchical Clustering {same as kmeans  but different algo.}

# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing mall dataset with pandas
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

# finding optimal number of clusters using dendrogram
import scipy.cluster.hierarchy as sch
# method used to find the clusters is ward method here.
# this ward method tries to minimize the variance in each cluster.
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.show()
# output: we can see that optimal no of clusters is 5 here.

# fitting hierarchical clustering to the model dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

# Visualizing the clusters(only for 2D clustering)
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('clusters of clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score[1-100]')
plt.legend()
plt.show()
# after execution we find that cluster 1 = careful, cluster2 = standard, cluster3 = target, cluster4=careless, cluster5=sensible.
# through this, we know that targeting cluster 3 clients will be most beneficial.