#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import hashlib
import shutil
import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from transformers import ViTModel, ViTImageProcessor


# In[6]:


model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")


# In[7]:


# Function to handle resizing
def resize_image(image, size=(224, 224)):
    """
    Resizes the image to the specified size.
    """
    return image.resize(size)


# In[8]:


# Function to convert image to greyscale
def convert_to_greyscale(image):
    """
    Converts the image to greyscale.
    """
    return image.convert("L")


# In[9]:


def remove_corrupt_image(image_path):
    """
    Attempts to open the image and returns True if valid, False if corrupt.
    """
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verifies if the image is corrupt
        return True
    except (OSError, ValueError):
        return False


# In[10]:


def process_image(image_path, new_path, size=(224, 224)):
    """
    Processes the image: resizing, converting to greyscale, and handling corrupt images.
    """
    if remove_corrupt_image(image_path):
        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")  # Convert to RGB (remove color profile)
                img = resize_image(img, size)  # Resize image
                img = convert_to_greyscale(img)  # Convert to greyscale
                img.save(new_path)  # Save processed image
                print(f"Processed and saved {image_path}")
        except Exception as e:
            print(f"Skipping {image_path}, could not process. Error: {e}")
    else:
        print(f"Skipping {image_path}, image is corrupt.")


# In[11]:


def process_images_in_directory(old_dir, new_dir, size=(224, 224)):
    """
    Processes all images in the given directory: resizing, greyscale conversion, and skipping corrupt ones.
    """
    os.makedirs(new_dir, exist_ok=True)  # Ensure the destination directory exists

    # Process all images in the old directory
    for filename in os.listdir(old_dir):
        if filename.endswith((".png", ".jpg", ".jpeg")):  # Filter image files
            old_path = os.path.join(old_dir, filename)
            new_path = os.path.join(new_dir, filename)
            process_image(old_path,  new_path, size)


# In[12]:


process_images_in_directory("old_favicons","resized_favicons")


# In[13]:


import torch
from torchvision import models, transforms
from PIL import Image

# Load pre-trained ResNet-50 model (without classification head)
model = models.resnet50(pretrained=True)
model = model.eval()  # Set to evaluation mode

# Remove the final fully connected layer (classification head)
model = torch.nn.Sequential(*list(model.children())[:-1])

# Define image transformation pipeline (resize, normalize, etc.)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(image_path):
    """Extract feature vector from an image."""
    img = Image.open(image_path).convert('RGB')  # Open image
    img = transform(img).unsqueeze(0)  # Apply transformations
    with torch.no_grad():
        features = model(img)  # Get the features from the model
    return features.flatten()  # Flatten the features to a 1D vector


# In[14]:


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
import os

def compute_similarity(features1, features2):
    """Compute cosine similarity between two feature vectors."""
    return cosine_similarity([features1], [features2])[0][0]

# Store features and their corresponding image paths
image_features = []
image_paths = []

# Extract features for all images with a progress bar
for filename in tqdm(os.listdir("resized_favicons"), desc="Extracting features", unit="image"):
    if filename.endswith((".png", ".jpg", ".jpeg")):
        image_path = os.path.join("resized_favicons", filename)
        features = extract_features(image_path)
        image_features.append(features)
        image_paths.append(image_path)

# Now, compare features to remove duplicates with a progress bar
threshold = 0.95  # Set a similarity threshold for near-duplicates
unique_images = []
seen_indices = set()

for i in tqdm(range(len(image_features)), desc="Comparing images", unit="image"):
    if i not in seen_indices:
        for j in range(i + 1, len(image_features)):
            similarity = compute_similarity(image_features[i], image_features[j])
            if similarity > threshold:
                seen_indices.add(j)  # Mark the duplicate image index
        unique_images.append(image_paths[i])

print(f"Found {len(unique_images)} unique images.")


# In[15]:


import os
import shutil

# Folder where unique images will be stored
unique_images_dir = "unique_images"

# Create the folder if it doesn't exist
os.makedirs(unique_images_dir, exist_ok=True)

# Copy the unique images to the new directory
for image_path in tqdm(unique_images, desc="Copying unique images", unit="image"):
    try:
        shutil.copy(image_path, os.path.join(unique_images_dir, os.path.basename(image_path)))
        print(f"Copied: {os.path.basename(image_path)}")
    except Exception as e:
        print(f"Error copying {os.path.basename(image_path)}: {e}")

print(f"All unique images are stored in '{unique_images_dir}'.")


# In[16]:


from sklearn.preprocessing import StandardScaler

# Step 1: Extract features for all unique images
image_features = []
image_paths = []

for image_path in tqdm(unique_images, desc="Extracting features", unit="image"):
    features = extract_features(image_path)
    image_features.append(features)
    image_paths.append(image_path)

# Convert to NumPy array
image_features = np.array(image_features)

# Step 2: Standardize features before clustering
scaler = StandardScaler()
features_scaled = scaler.fit_transform(image_features)


# In[17]:


from sklearn.cluster import KMeans
import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Extract features for unique images
features = [extract_features(path) for path in unique_images]

# Convert to numpy array for clustering
features_array = np.array(features)


# In[21]:


from sklearn.decomposition import PCA

# Reduce to 50 dimensions, or even 2 for t-SNE/visualization
# pca = PCA(n_components=50)
# features_pca = pca.fit_transform(features_array)

# Now cluster again with KMeans
kmeans = KMeans(n_clusters=50, random_state=42)
labels = kmeans.fit_predict(features_array)


# In[ ]:


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
visualize_kmeans_clusters(unique_images, labels)


# In[20]:


from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Use the same features used for clustering
features_array = np.array(features)

# Silhouette Score
sil_score = silhouette_score(features_array, labels)
print(f"Silhouette Score: {sil_score:.4f}")

# Calinski-Harabasz Index
ch_score = calinski_harabasz_score(features_array, labels)
print(f"Calinski-Harabasz Index: {ch_score:.4f}")

# Davies-Bouldin Index
db_score = davies_bouldin_score(features_array, labels)
print(f"Davies-Bouldin Index: {db_score:.4f}")

