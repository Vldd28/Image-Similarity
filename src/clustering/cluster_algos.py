# Now cluster again with KMeans
from sklearn.cluster import KMeans,DBSCAN

def KMEANS(features,n_clusters=50):
    kmeans = KMeans(n_clusters, random_state=42)
    labels = kmeans.fit_predict(features)
    return labels

def DBSCAN(features,eps):
    return DBSCAN(eps).labels