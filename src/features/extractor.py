from PIL import Image
import torch
from torchvision import models, transforms
import tqdm
def extract_features(image_path,pipeline,model):
    """Extract feature vector from an image."""
    img = Image.open(image_path).convert('RGB')  # Open image
    img = pipeline(img).unsqueeze(0)  # Apply transformations
    with torch.no_grad():
        features = model(img)  # Get the features from the model
    return features.flatten()  # Flatten the features to a 1D vector
# Extract features for all images with a progress bar
def feature_extraction():
    image_features = []
    image_paths = []
    for filename in tqdm(os.listdir("resized_favicons"), desc="Extracting features", unit="image"):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join("resized_favicons", filename)
            features = extract_features(image_path)
            image_features.append(features)
            image_paths.append(image_path)
