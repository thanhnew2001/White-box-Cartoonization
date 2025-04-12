import cv2
import numpy as np
import requests
import tensorflow as tf
from huggingface_hub import snapshot_download
from PIL import Image
import os
import argparse
import glob
from tqdm import tqdm
import subprocess
import mimetypes

# Force CPU-only operations
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

# Disable GPU
tf.config.set_visible_devices([], 'GPU')

def resize_crop(image):
    h, w, c = np.shape(image)
    if min(h, w) > 720:
        if h > w:
            h, w = int(720 * h / w), 720
        else:
            h, w = 720, int(720 * w / h)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    h, w = (h // 8) * 8, (w // 8) * 8
    image = image[:h, :w, :]
    return image

def load_image(image_path):
    """Load image from local path or URL"""
    if image_path.startswith(('http://', 'https://')):
        # Handle URL
        image = Image.open(requests.get(image_path, stream=True).raw)
        image = image.convert("RGB")
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        # Handle local file
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
    return image

def preprocess_image(image):
    image = resize_crop(image)
    image = image.astype(np.float32) / 127.5 - 1
    image = np.expand_dims(image, axis=0)
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    return image

def process_single_image(image_path, output_path, concrete_func):
    """Process a single image using the loaded model"""
    try:
        # Load and preprocess image
        image = load_image(image_path)
        preprocessed_image = preprocess_image(image)

        # Run inference with CPU device placement
        with tf.device('/CPU:0'):
            result = concrete_func(preprocessed_image)["final_output:0"]

        # Post-process the result
        output = (result[0].numpy() + 1.0) * 127.5
        output = np.clip(output, 0, 255).astype(np.uint8)
        
        # Save the result
        cv2.imwrite(output_path, output)
        return True
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return False

def process_folder(input_path, output_path):
    """Process all images in a folder"""
    # Load the model once for all images
    try:
        model_path = snapshot_download("sayakpaul/whitebox-cartoonizer")
        loaded_model = tf.saved_model.load(model_path)
        concrete_func = loaded_model.signatures["serving_default"]
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Get all image files
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', 
                       '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.TIFF')
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_path, ext)))
    
    if not image_files:
        raise ValueError(f"No image files found in {input_path}")

    # Process each image
    print(f"Processing {len(image_files)} images...")
    successful = 0
    failed = 0

    for image_file in tqdm(image_files):
        output_file = os.path.join(output_path, os.path.basename(image_file))
        if process_single_image(image_file, output_file, concrete_func):
            successful += 1
        else:
            failed += 1

    print(f"\nProcessing complete:")
    print(f"Successfully processed: {successful} images")
    if failed > 0:
        print(f"Failed to process: {failed} images")
    print(f"Output saved to: {output_path}")

def process_video(input_path, output_path):
    """Process video file using FFmpeg for frame extraction and merging"""
    # Create temp directory for frames
    temp_dir = os.path.join(os.path.dirname(output_path), "temp_frames")
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Load the model once for all frames
        model_path = snapshot_download("sayakpaul/whitebox-cartoonizer")
        loaded_model = tf.saved_model.load(model_path)
        concrete_func = loaded_model.signatures["serving_default"]
        
        # Extract frames using FFmpeg (exactly 8 fps)
        print("Extracting frames...")
        frame_pattern = os.path.join(temp_dir, 'frame_%07d.png')
        
        # Extract frames at exactly 8fps using precise filtering
        subprocess.run([
            'ffmpeg', '-i', input_path,
            '-loglevel', 'error',
            '-vf', 'fps=8,scale=trunc(iw/2)*2:trunc(ih/2)*2',  # Ensure even dimensions
            '-frame_pts', '1',
            '-q:v', '2',  # High quality frames
            frame_pattern
        ], check=True)
        
        # Process frames
        print("Processing frames...")
        frame_files = sorted([f for f in os.listdir(temp_dir) if f.startswith('frame_')])
        
        for frame_file in tqdm(frame_files):
            frame_path = os.path.join(temp_dir, frame_file)
            if not process_single_image(frame_path, frame_path, concrete_func):
                print(f"Warning: Failed to process frame {frame_file}")
        
        # Verify processed frames exist
        processed_frames = sorted([f for f in os.listdir(temp_dir) if f.startswith('frame_')])
        if not processed_frames:
            raise RuntimeError("No processed frames found")
        
        # Combine frames back into video with original audio
        print("Creating output video...")
        subprocess.run([
            'ffmpeg', '-y',
            '-r', '8',  # Force 8 fps input
            '-i', frame_pattern,
            '-i', input_path,
            '-map', '0:v',
            '-map', '1:a?',
            '-c:v', 'libx264',
            '-preset', 'veryfast',  # Faster encoding
            '-tune', 'animation',   # Optimize for animated content
            '-crf', '23',          # Good quality
            '-pix_fmt', 'yuv420p',
            '-g', '8',             # GOP size matches fps
            '-threads', '0',        # Use all available CPU threads
            '-loglevel', 'error',
            output_path
        ], check=True)
        
        print(f"Video processing complete. Output saved to: {output_path}")
        print(f"Temporary frames saved in: {temp_dir}")
        
    except Exception as e:
        print(f"Error during video processing: {str(e)}")
        raise
    finally:
        # Keep temp directory for tracking
        pass

def process_input(input_path, output_path):
    """Process input which can be an image, video, URL, or folder"""
    if os.path.isdir(input_path):
        process_folder(input_path, output_path)
    else:
        # For single file, ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Check if input is a video
        if os.path.isfile(input_path) and mimetypes.guess_type(input_path)[0].startswith('video'):
            process_video(input_path, output_path)
        else:
            # Load model and process single image
            model_path = snapshot_download("sayakpaul/whitebox-cartoonizer")
            loaded_model = tf.saved_model.load(model_path)
            concrete_func = loaded_model.signatures["serving_default"]
            
            if process_single_image(input_path, output_path, concrete_func):
                print(f"Successfully processed image: {output_path}")
            else:
                print("Failed to process image")

def main():
    parser = argparse.ArgumentParser(description='Image/Video Cartoonization using Hugging Face Model')
    parser.add_argument('--input', type=str, required=True, 
                      help='Path to input image, video, URL, or folder containing images')
    parser.add_argument('--output', type=str, required=True, 
                      help='Path to save the output image/video or folder')
    args = parser.parse_args()
    
    process_input(args.input, args.output)

if __name__ == "__main__":
    main()
