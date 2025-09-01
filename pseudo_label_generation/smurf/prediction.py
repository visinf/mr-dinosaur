import os
import torch
import torch.nn as nn
import torchvision
import pandas as pd
import numpy as np
import json
from torch import Tensor
from torchvision.utils import flow_to_image
from smurf.smurf import raft_smurf
from typing import List
from torchvision import transforms
from PIL import Image
import time


def process_optical_flow(
    image1: Tensor,
    image2: Tensor,
    output_img_dir: str,
    output_csv_dir: str,
    model: nn.Module,
    image1_path: str,
    npy_output_dir: str,
) -> None:
    """
    Process a pair of images to compute optical flow, save the flow image, CSV, and flow as a numpy array.
    """
    # global pair_idx
    # Get the scene and base filename from the input image path
    scene_name = os.path.basename(os.path.dirname(image1_path))  # Extract the scene folder name
    filename = os.path.basename(image1_path)  # Extract the base filename
    # Check if output files already exist
    flow_image_dir = os.path.join(output_img_dir, scene_name)
    flow_image_path = os.path.join(flow_image_dir, filename)
    flow_csv_dir = os.path.join(output_csv_dir, scene_name)
    flow_csv_path = os.path.join(flow_csv_dir, filename.replace(".png", ".csv"))
    npy_file_dir = os.path.join(npy_output_dir, scene_name)
    npy_file_path = os.path.join(npy_file_dir, filename.replace(".png", ".npy"))

    # If output files exist, skip processing
    if (
        os.path.exists(flow_image_path)
        and os.path.exists(flow_csv_path)
        and os.path.exists(npy_file_path)
    ):
        print(f"Skipping {filename}, already processed.")
        return filename, None, None

    # Move images to GPU
    image1 = image1.cuda()
    image2 = image2.cuda()

    with torch.no_grad():
        optical_flow: List[Tensor] = model(image1[None], image2[None])
    flow_image = flow_to_image(
        optical_flow[-1][0].cpu()
    )  # Convert flow to RGB image and move to CPU

    # Convert optical flow from (2, H, W) to (H, W, 2)
    flow_np = optical_flow[-1][0].detach().cpu().numpy().transpose(1, 2, 0)  # Shape: (H, W, 2)

    # Compute the magnitude of optical flow
    flow_magnitude = (
        torch.sqrt(optical_flow[-1][0][0] ** 2 + optical_flow[-1][0][1] ** 2).detach().cpu().numpy()
    )

    scene_name = os.path.basename(os.path.dirname(image1_path))  # Extract the scene folder name
    filename = os.path.basename(image1_path)  # Extract the base filename

    # Save the optical flow image
    flow_image_dir = os.path.join(output_img_dir, scene_name)
    os.makedirs(flow_image_dir, exist_ok=True)
    flow_image_path = os.path.join(flow_image_dir, filename)
    torchvision.io.write_png(flow_image, flow_image_path)

    # Save the magnitude of the optical flow as a CSV
    flow_csv_dir = os.path.join(output_csv_dir, scene_name)
    os.makedirs(flow_csv_dir, exist_ok=True)
    flow_csv_path = os.path.join(flow_csv_dir, filename.replace(".png", ".csv"))
    df = pd.DataFrame(flow_magnitude)
    df.to_csv(flow_csv_path, index=False, header=False, float_format="%.3f")

    # Save the optical flow as a numpy array
    npy_file_dir = os.path.join(npy_output_dir, scene_name)
    os.makedirs(npy_file_dir, exist_ok=True)
    npy_file_path = os.path.join(npy_file_dir, filename.replace(".png", ".npy"))
    np.save(npy_file_path, flow_np)

    filename = filename.replace(".png", "")

    return filename, flow_magnitude, scene_name


def calculate_corner_averages(flow_magnitude: np.ndarray, corner_size: float = 0.15) -> dict:
    """
    Calculate average optical flow magnitude in four corners of the flow magnitude array.
    """
    height, width = flow_magnitude.shape
    corner_height = int(height * corner_size)
    corner_width = int(width * corner_size)

    corners = {
        "top_left": flow_magnitude[:corner_height, :corner_width],
        "top_right": flow_magnitude[:corner_height, -corner_width:],
        "bottom_left": flow_magnitude[-corner_height:, :corner_width],
        "bottom_right": flow_magnitude[-corner_height:, -corner_width:],
    }
    return {corner: float(np.mean(corners[corner])) for corner in corners}


def process_and_save_optical_flow(
    base_dir: str,
    output_img_dir: str,
    output_csv_dir: str,
    npy_output_dir: str,
    corner_size: float = 0.15,
    checkpoint: str = "smurf_kitti.pt",
) -> None:
    """
    Process images in the given directory to compute optical flow and save flow values as CSV, flow images, numpy arrays, and a JSON summary.
    """
    # Ensure output directories exist
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_csv_dir, exist_ok=True)
    os.makedirs(npy_output_dir, exist_ok=True)

    # Load the model and move it to GPU
    model = raft_smurf(
        checkpoint=checkpoint,  # Path to the pre-trained model checkpoint
    )
    model = model.cuda()
    model.eval()  # Set the model to evaluation mode
    # Traverse directory and process image pairs
    for subdir, dirs, files in os.walk(base_dir):
        image_pairs = []
        # Collect consecutive image pairs
        files = sorted([f for f in files if f.endswith(".png")])
        print("Processing image pairs...")
        for i in range(len(files) - 1):
            image1_path = os.path.join(subdir, files[i])
            image2_path = os.path.join(subdir, files[i + 1])
            image_pairs.append((image1_path, image2_path))

        scene_data_dict = {}  # Initialize the dictionary for scene-specific data
        # Process each image pair sequentially (single thread)
        for image1_path, image2_path in image_pairs:
            image1: Tensor = torchvision.io.read_image(
                image1_path, mode=torchvision.io.ImageReadMode.RGB
            )
            image2: Tensor = torchvision.io.read_image(
                image2_path, mode=torchvision.io.ImageReadMode.RGB
            )
            # Normalize images to range [-1, 1]
            image1 = 2.0 * (image1 / 255.0) - 1.0
            image2 = 2.0 * (image2 / 255.0) - 1.0
            start = time.perf_counter()  # ① 开始计时
            # Process the optical flow for this pair
            filename, flow_magnitude, scene_name = process_optical_flow(
                image1, image2, output_img_dir, output_csv_dir, model, image1_path, npy_output_dir
            )
            if flow_magnitude is None:
                continue
            # Compute the corner averages
            corner_averages = calculate_corner_averages(flow_magnitude, corner_size)

            # Add the result to the dictionary for the specific scene
            if scene_name not in scene_data_dict:
                scene_data_dict[scene_name] = {}

            scene_data_dict[scene_name][filename] = corner_averages

        # After processing all pairs in the scene, save the JSON file for this scene
        for scene_name, corner_data in scene_data_dict.items():
            json_file_dir = os.path.join(output_csv_dir, scene_name)
            os.makedirs(json_file_dir, exist_ok=True)
            json_file_path = os.path.join(json_file_dir, f"{scene_name}.json")

            with open(json_file_path, "w") as json_file:
                json.dump(corner_data, json_file, indent=4)

        # Clear the data dictionary for the scene to process the next scene
        scene_data_dict.clear()


