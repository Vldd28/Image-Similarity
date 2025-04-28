import os
import numpy as np
import torch
import faiss
from PIL import Image
from transformers import ViTModel, ViTImageProcessor
from data import preprocessing as preproc
from features import extractor as extract
from clustering import cluster_algos,visualize
from utils import helper 
# from torchvision.models import resnet50, ResNet50_Weights
# preproc.process_images_in_directory("../old_favicons","../resized_favicons")

# from torchvision import models, transforms

# # Load pre-trained ResNet-50 model
# model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
# model = model.eval()

# # Remove the final fully connected layer (classification head)
# model = torch.nn.Sequential(*list(model.children())[:-1])

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# image_features,image_paths = extract.feature_extraction("../resized_favicons",transform, model)

# image_features_np = np.array(image_features)

# # Save the features and paths locally
# np.save("../saved_features/image_features.npy", image_features_np)
# np.save("../saved_features/image_paths.npy", image_paths)

image_features = np.load("../saved_features/image_features.npy")
image_paths = np.load("../saved_features/image_paths.npy", allow_pickle=True)

labels = cluster_algos.DBSCAN_clustering(image_features)
helper.save_cluster_images(image_paths,labels)
# visualize.visualize_kmeans_clusters(image_paths,labels)
