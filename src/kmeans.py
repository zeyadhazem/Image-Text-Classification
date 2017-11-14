from sklearn.cluster import KMeans
import numpy as np

data = np.array([[0,0], [1,1], [2,2], [3,0]])

kmeans = KMeans(n_clusters=3, random_state=0).fit(data)

print (kmeans.cluster_centers_)
