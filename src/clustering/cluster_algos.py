# Now cluster again with KMeans
from sklearn.cluster import KMeans,DBSCAN

def KMEANS(features,n_clusters=50):
    kmeans = KMeans(n_clusters, random_state=42)
    labels = kmeans.fit_predict(features)
    return labels

from sklearn.cluster import DBSCAN

def DBSCAN_clustering(features, eps=0.3, min_samples=5):
    # Instantiate DBSCAN model with given parameters
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    
    # Fit the DBSCAN model and return the cluster labels
    labels = dbscan.fit_predict(features)
    return labels
