import os
import numpy as np
import torch
import faiss
from PIL import Image
from transformers import ViTModel, ViTImageProcessor

# Load ViT model and processor
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

# Function to get ViT embedding
def get_vit_embedding(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

        if embedding.shape[0] == 0:
            print(f"Warning: Empty embedding for {image_path}")
        return embedding

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Directory of favicon images
logo_dir = "resized_favicons"
logo_paths = [os.path.join(logo_dir, f) for f in os.listdir(logo_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

# Get embeddings
logo_embeddings = []
valid_logo_paths = []

for logo_path in logo_paths:
    embedding = get_vit_embedding(logo_path)
    if embedding is not None:
        logo_embeddings.append(embedding)
        valid_logo_paths.append(logo_path)

logo_embeddings = np.array(logo_embeddings).astype('float32')

# Create FAISS index
dimension = logo_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(logo_embeddings)

print(f"Indexed {len(logo_embeddings)} favicon embeddings.")