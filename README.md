# Favicon Deduplication and Clustering Project

## Overview

This project focuses on deduplicating, processing, and clustering a large collection of website favicon images. The goal is to group visually similar favicons together while removing near-identical duplicates, enabling further analysis such as identifying trends, detecting counterfeit branding, and assisting cybersecurity efforts.

## Motivation and Applications

- **Brand Monitoring**: Companies can monitor if their logo or favicon is being misused across the internet.
- **Cybersecurity and Phishing Detection**: Many phishing websites copy or slightly modify the favicons of legitimate websites to deceive users. By clustering similar favicons and detecting duplicates or near-duplicates, this system can help identify phishing sites that attempt to impersonate trusted brands.
- **Search and Organization**: Improves the organization and searchability of large favicon datasets.

## How It Works

1. **Preprocessing**:
   - Resize all favicons to a standard size (224x224).
   - Convert images to greyscale for consistency and to reduce noise.
   - Remove corrupt or unreadable images.

2. **Feature Extraction**:
   - Use a pre-trained ResNet-50 model (without its classification head) to extract meaningful feature vectors from each favicon.

3. **Deduplication**:
   - Compute cosine similarity between all pairs of favicon feature vectors.
   - Filter out near-duplicate favicons based on a similarity threshold (e.g., 0.95).

4. **Clustering**:
   - Standardize the extracted features.
   - Apply KMeans clustering to group similar favicons.
   - Visualize the resulting clusters.

5. **Evaluation**:
   - Metrics like Silhouette Score, Calinski-Harabasz Index, and Davies-Bouldin Index are used to assess clustering performance.

## Technologies Used

- **Python 3.8+**
- **PyTorch**
- **Torchvision**
- **Scikit-learn**
- **PIL (Pillow)**
- **OpenCV**
- **Matplotlib**
- **Transformers (Huggingface)**

## Project Structure

```
old_favicons/         # Original favicons
resized_favicons/     # Preprocessed favicons (resized and greyscale)
unique_images/        # Deduplicated favicons
scripts/              # Python scripts for processing and clustering
README.md             # Project documentation
```

## Setup Instructions

1. **Clone the repository**:

```bash
git clone https://github.com/yourusername/favicon-clustering.git
cd favicon-clustering
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Run processing and clustering**:

Make sure your original favicons are placed inside the `old_favicons/` folder. Then run the provided scripts to preprocess, deduplicate, and cluster the images.

## Future Work

- Fine-tuning deep learning models specifically for favicon feature extraction.
- Integrating clustering results into a live phishing detection pipeline.
- Exploring more advanced clustering techniques (e.g., DBSCAN, Hierarchical Clustering).
- Real-time favicon lookup service to check for impersonations.

## License

MIT License

---

Feel free to contribute, raise issues, or suggest improvements!

