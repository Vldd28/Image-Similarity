from PIL import Image
import torch
from torchvision import models, transforms
import tqdm
import os

def corrupt_image(image_path):
    """
    Attempts to open the image and returns True if valid, False if corrupt.
    """
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verifies if the image is corrupt
        return True
    except (OSError, ValueError):
        return False
    
def extract_features(image_path,pipeline,model):
    """Extract feature vector from an image."""
    if(corrupt_image(image_path) == True):
        img = Image.open(image_path).convert('RGB')  # Open image
        img = pipeline(img).unsqueeze(0)  # Apply transformations
        with torch.no_grad():
            features = model(img)  # Get the features from the model
        return features.flatten()  # Flatten the features to a 1D vector
# Extract features for all images with a progress bar

def feature_extraction(folder, pipeline, model):
    image_features = []
    image_paths = []
    for filename in tqdm.tqdm(os.listdir(folder), desc="Extracting features", unit="image"):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(folder, filename)
            features = extract_features(image_path,pipeline,model)
            image_features.append(features)
            image_paths.append(image_path)
    return image_features,image_paths
