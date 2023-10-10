import cv2
import os

# Directory containing the input images
input_dir = 'frames_from_videos_hello'

# Directory to save the preprocessed images
output_dir = 'preprocessed_images'
os.makedirs(output_dir, exist_ok=True)


# Function to preprocess an image
def preprocess_image(image_path, output_path, target_size=(224, 224)):
    try:
        # Read the image
        image = cv2.imread(image_path)

        # Ensure the image was read successfully
        if image is None:
            print(f"Skipping {image_path} as it could not be read.")
            return

        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize to the target size
        resized_image = cv2.resize(gray_image, target_size)

        # Normalize pixel values to the range [0, 1]
        normalized_image = resized_image / 255.0

        # Save the preprocessed image
        cv2.imwrite(output_path, normalized_image * 255.0)  # Save as 8-bit image

    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")


# Iterate through all image files in the input directory
for root, _, files in os.walk(input_dir):
    for filename in files:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            input_image_path = os.path.join(root, filename)
            output_image_path = os.path.join(output_dir, filename)
            preprocess_image(input_image_path, output_image_path)

print("Image preprocessing complete.")
