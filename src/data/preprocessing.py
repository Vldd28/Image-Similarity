from PIL import Image
import os
# Function to handle resizing
def resize_image(image, size=(224, 224)):
    """
    Resizes the image to the specified size.
    """
    return image.resize(size)

# Function to convert image to greyscale
def convert_to_greyscale(image):
    """
    Converts the image to greyscale.
    """
    return image.convert("L")

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

def process_image(image_path, new_path, size=(224, 224)):
    """
    Processes the image: resizing, converting to greyscale, and handling corrupt images.
    """
    if remove_corrupt_image(image_path):
        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")  # Convert to RGB (remove color profile)
                img = resize_image(img, size)  # Resize image
                # img = convert_to_greyscale(img)  # Convert to greyscale
                img.save(new_path)  # Save processed image
                print(f"Processed and saved {image_path}")
        except Exception as e:
            print(f"Skipping {image_path}, could not process. Error: {e}")
    else:
        print(f"Skipping {image_path}, image is corrupt.")

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

