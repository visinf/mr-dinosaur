import os
import glob
import cv2
import subprocess
from concurrent.futures import ThreadPoolExecutor


import os
import cv2
from concurrent.futures import ThreadPoolExecutor

def resize_image(image_path, output_size):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    output_width, output_height = output_size

    aspect_ratio = width / height
    output_aspect_ratio = output_width / output_height

    if aspect_ratio > output_aspect_ratio:
        new_height = output_height
        new_width = int(new_height * aspect_ratio)
    else:
        new_width = output_width
        new_height = int(new_width / aspect_ratio)

    resized_image = cv2.resize(image, (new_width, new_height))

    cv2.imwrite(image_path, resized_image)

def resize_image_worker(image_path, output_size):
    """
    Worker function to resize a single image.
    """
    resize_image(image_path, output_size)

def resize_images(directory, output_size, num_workers=8):
    """
    Resize all PNG images in a directory and its subdirectories using multiple workers.
    
    Parameters:
    directory (str): The directory containing the images.
    output_size (tuple): The target size for resizing the images.
    num_workers (int): Number of worker threads for concurrent image resizing.
        """
    # Create a list to store all image paths
    image_paths = []

    # Traverse the directory and find all PNG images
    for root, dirs, files in os.walk(directory):
        print(f"Resizing in {root}")
        for file in files:
            if file.endswith(".png"):
                image_path = os.path.join(root, file)
                image_paths.append(image_path)

    # Use ThreadPoolExecutor to process the images concurrently
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(resize_image_worker, image_path, output_size) for image_path in image_paths]

        # Wait for all the futures to complete
        for future in futures:
            future.result()



def process_images(data_path,rgb_path='png_image', model_path='raft-kitti.pth', reverse_flag=True, image_res=(0,0)):
    print("Computing Optical Flow ......")
    rgbpath = os.path.join(data_path, rgb_path)
    if not os.path.exists(rgbpath):
        print(f"Error: The directory {rgbpath} does not exist.")
        return
    if image_res!=(0,0):
        resize_images(rgbpath, image_res)

    gap = [1]
    reverse = [0, 1] if reverse_flag else [0]
    folder = glob.glob(os.path.join(rgbpath, '*'))

    for f in folder:
        for r in reverse:
            for g in gap:
                print(f'===> Running {f}, gap {g}')
                mode = model_path
                if r == 1:
                    raw_outroot = os.path.join(data_path, f'Flows_gap-{g}/')
                    outroot = os.path.join(data_path, f'FlowImages_gap-{g}/')
                else:
                    raw_outroot = os.path.join(data_path, f'Flows_gap{g}/')
                    outroot = os.path.join(data_path, f'FlowImages_gap{g}/')

                command = [
                    "python", "raft/predict.py",
                    "--gap", str(g),
                    "--model", mode,
                    "--path", f,
                    "--outroot", outroot,
                    "--reverse", str(r),
                    "--raw_outroot", raw_outroot
                ]

                # Execute the command
                try:
                    result = subprocess.run(command, check=True, capture_output=True, text=True)
                    print(f"Output for {f}, gap {g}:\n{result.stdout}")
                except subprocess.CalledProcessError as e:
                    print(f"Error executing command for {f}, gap {g}:\n{e.stderr}")
                    
