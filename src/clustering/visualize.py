import matplotlib as plt
import matplotlib.image as mpimg

def visualize_kmeans_clusters(image_paths, labels, max_per_cluster=10):
    clusters = {}
    for path, label in zip(image_paths, labels):
        clusters.setdefault(label, []).append(path)

    for cluster_id, paths in clusters.items():
        plt.figure(figsize=(15, 3))
        plt.suptitle(f"Cluster {cluster_id} ({len(paths)} images)", fontsize=16)
        for i, image_path in enumerate(paths[:max_per_cluster]):
            plt.subplot(1, min(len(paths), max_per_cluster), i + 1)
            img = mpimg.imread(image_path)
            plt.imshow(img)
            plt.axis('off')
        plt.tight_layout()
        plt.show()
