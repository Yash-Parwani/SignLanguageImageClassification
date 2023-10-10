import cv2
import os

# Directory containing the video files
video_dir = 'Data/hello'

# Create a directory to save the extracted frames
output_dir = 'frames_from_videos_hello'
os.makedirs(output_dir, exist_ok=True)

# Function to extract frames from a video
def extract_frames(video_path, output_path):
    # Initialize the video capture object
    cap = cv2.VideoCapture(video_path)

    # Initialize a frame counter
    frame_count = 0

    while True:
        # Read the next frame from the video
        ret, frame = cap.read()

        # If the frame could not be read, we've reached the end of the video
        if not ret:
            break

        # Define the path to save the frame as an image
        frame_filename = f'{os.path.splitext(os.path.basename(video_path))[0]}_frame_{frame_count:04d}.jpg'
        output_frame_path = os.path.join(output_path, frame_filename)

        # Save the frame as an image
        cv2.imwrite(output_frame_path, frame)

        # Increment the frame counter
        frame_count += 1

    # Release the video capture object
    cap.release()

    print(f"Extracted {frame_count} frames from {video_path}")

# Iterate through all video files in the directory
for root, _, files in os.walk(video_dir):
    for filename in files:
        if filename.endswith('.mp4'):
            video_path = os.path.join(root, filename)
            extract_frames(video_path, output_dir)
