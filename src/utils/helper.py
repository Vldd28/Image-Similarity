import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def save_cluster_images(image_paths, labels, output_dir="../clusters"):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    clusters = {}
    for path, label in zip(image_paths, labels):
        clusters.setdefault(label, []).append(path)
    
    # Loop over each cluster and save its images to its respective folder
    for cluster_id, (cluster_label, paths) in enumerate(clusters.items()):
        cluster_folder = os.path.join(output_dir, f"cluster_{cluster_label}")
        
        # Create folder for the cluster if it doesn't exist
        if not os.path.exists(cluster_folder):
            os.makedirs(cluster_folder)
        
        for i, image_path in enumerate(paths):
            # Read the image
            img = mpimg.imread(image_path)
            
            # Create the path for the new image
            image_name = os.path.basename(image_path)
            save_path = os.path.join(cluster_folder, f"{image_name}")
            
            # Save the image
            plt.imsave(save_path, img)
            print(f"Saved image {image_name} in {cluster_folder}")

    print(f"All clusters' images have been saved in {output_dir}")
