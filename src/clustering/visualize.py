import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def visualize_kmeans_clusters(image_paths, labels, max_per_cluster=10, output_file="kmeans_all_clusters.png"):
    clusters = {}
    for path, label in zip(image_paths, labels):
        clusters.setdefault(label, []).append(path)
    
    # Calculate the size of the final image (height will depend on number of clusters and max images per cluster)
    num_clusters = len(clusters)
    rows = num_clusters
    max_images_in_cluster = max_per_cluster

    # Set up a big figure with enough space for all clusters
    fig, axes = plt.subplots(nrows=rows, ncols=max_images_in_cluster, figsize=(15, 3 * rows))

    # Loop over each cluster
    for cluster_id, (cluster_label, paths) in enumerate(clusters.items()):
        for i, image_path in enumerate(paths[:max_images_in_cluster]):
            ax = axes[cluster_id, i]  # Get the axis for the current image
            img = mpimg.imread(image_path)
            ax.imshow(img)
            ax.axis('off')  # Hide axes
        axes[cluster_id, 0].set_title(f"Cluster {cluster_label} ({len(paths)} images)", fontsize=16)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)  # Adjust vertical spacing between rows
    plt.savefig(output_file)
    print(f"Saved all clusters in {output_file}")

# Example usage
# visualize_kmeans_clusters(image_paths, labels)
