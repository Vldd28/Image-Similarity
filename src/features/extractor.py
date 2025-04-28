from PIL import Image
import torch
from torchvision import models, transforms
import tqdm
import os
import numpy as np
def corrupt_image(image_path):
    """
    Attempts to open the image and returns True if valid, False if corrupt.
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except (OSError, ValueError):
        return False

def extract_features(image_path, pipeline, model):
    """Extract feature vector from an image."""
    if corrupt_image(image_path):
        img = Image.open(image_path).convert('RGB')
        img = pipeline(img).unsqueeze(0)
        with torch.no_grad():
            features = model(img)  # Still a torch tensor
        return features.flatten().cpu().numpy()  # Convert to NumPy array immediately
    else:
        return None

def feature_extraction(folder, pipeline, model):
    image_features = []
    image_paths = []
    for filename in tqdm.tqdm(os.listdir(folder), desc="Extracting features", unit="image"):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(folder, filename)
            features = extract_features(image_path, pipeline, model)
            if features is not None:
                image_features.append(features)
                image_paths.append(image_path)
    return np.array(image_features), image_paths  # Return numpy array directly
