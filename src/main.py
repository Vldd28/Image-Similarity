import os
import numpy as np
import torch
import faiss
from PIL import Image
from transformers import ViTModel, ViTImageProcessor
from data import preprocessing as preproc
from features import extractor as extract
from clustering import cluster_algos,visualize


# preproc.process_images_in_directory("../old_favicons","../resized_favicons")

from torchvision import models, transforms

# Load pre-trained ResNet-50 model (without classification head)
model = models.resnet50(pretrained=True)
model = model.eval()  
# my tool Set to evaluation mode

# Remove the final fully connected layer (classification head)
model = torch.nn.Sequential(*list(model.children())[:-1])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image_features,image_paths = extract.feature_extraction("../resized_favicons",transform, model)

labels = cluster_algos.KMEANS(image_features)

visualize.visualize_kmeans_clusters(image_paths,labels)
