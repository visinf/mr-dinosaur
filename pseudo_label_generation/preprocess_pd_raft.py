import os
import argparse
from utils import subset_selection_of as sof, data, binary_map2_instances
from raft import run_inference
import cv2
import math
from tqdm import tqdm  
import glob

def create_output_directories(*directories):
    """Create multiple directories from a list."""
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def process_images_diod_pd(base_input_dir, rgb_path):
    """
    Resize and crop all PNG images in the specified directory and its subdirectories.

    Args:
        base_input_dir (str): The base input directory.
        rgb_path (str): The subdirectory containing RGB images.
    """
    input_dir = os.path.join(base_input_dir, rgb_path)

    # Get all PNG images in the directory and subdirectories
    png_images = glob.glob(os.path.join(input_dir, '**', '*.png'), recursive=True)
    total_images = len(png_images)

    if total_images == 0:
        print(f"No PNG images found in {input_dir}.")
        return

    # Initialize tqdm progress bar
    with tqdm(total=total_images, desc="Processing images", unit="image") as pbar:
        for image_path in png_images:
            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Unable to read image {image_path}. Skipping.")
                pbar.update(1)
                continue

            # Define downsampling ratio and crop size
            downsampling_ratio = 0.5
            crop = 128

            # Calculate new dimensions
            width = int(math.ceil(image.shape[1] * downsampling_ratio))
            height = int(math.ceil(image.shape[0] * downsampling_ratio))
            dim = (width, height)

            try:
                # Resize the image
                resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)

                # Ensure the image has enough height to crop
                if resized_image.shape[0] <= crop:
                    print(f"Warning: Image height after resizing is too small to crop {image_path}. Skipping.")
                    pbar.update(1)
                    continue

                # Crop the top `crop` pixels
                cropped_image = resized_image[crop:, :, :]

                # Save the processed image back to the same path or a different path
                cv2.imwrite(image_path, cropped_image)
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
            finally:
                pbar.update(1)

def main(args):
    base_input_dir = os.path.join(args.base_dir, args.base_input_dir)
    Flows_gap1 = os.path.join(base_input_dir, "Flows_gap1")
    FlowsImages_gap1 = os.path.join(base_input_dir, "FlowImages_gap1")
    Flows_value_gap1 = os.path.join(base_input_dir, "Flows_value_gap1")
    base_of_output_dir = os.path.join(args.base_dir, f"PD_{args.optical_threshold}_{args.rgb_path}")
    if args.diod_pd:
        print("Processing images with diod_pd flag enabled...")
        process_images_diod_pd(base_input_dir, args.rgb_path)
        print("Image processing with diod_pd completed.")
    if args.compute_flow:
        run_inference.process_images(base_input_dir,args.rgb_path,args.model_path,args.reverse_flag)
        data.process_and_save_corner_flows(Flows_gap1, Flows_value_gap1, args.corner_size)
         
    sof.filter_and_save_images_by_corner_flow(base_input_dir, args.rgb_path,base_of_output_dir,"Flows_value_gap1", args.optical_threshold, args.sequence_length, args.required_corners)
    sof.sync_optical_flow_images(FlowsImages_gap1,os.path.join(base_of_output_dir, "optical_flow"),os.path.join(base_of_output_dir, "optical_flow_image"))
    sof.sync_optical_flow(Flows_gap1,os.path.join(base_of_output_dir, "optical_flow"),os.path.join(base_of_output_dir, "Flows_gap1"))

    data.recursive_csv_to_png(os.path.join(base_of_output_dir, "optical_flow"), base_of_output_dir, args.pixel_threshold)
    data.filter_and_delete_images(os.path.join(base_of_output_dir, f"init_seg_{args.pixel_threshold}"))
    binary_map2_instances.process_folder_files(base_of_output_dir,args.pixel_threshold)
    data.batch_merge_pngs(os.path.join(base_of_output_dir, f'instances_{args.pixel_threshold}'),os.path.join(base_of_output_dir, f"init_seg_{args.pixel_threshold}"), os.path.join(base_of_output_dir, "optical_flow_image"), os.path.join(base_of_output_dir, args.rgb_path), os.path.join(base_of_output_dir, f"combination_{args.pixel_threshold}"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process TRI-PD Dataset.")
    parser.add_argument("--base_dir",type=str,
        default="../scripts/datasets/data/TRI-PD",
        help="Base directory for processing and output",
    )
    parser.add_argument(
        "--base_input_dir", type=str, default="PD_simplified/camera_01", help="Base input directory",
    )
    parser.add_argument("--rgb_path", type=str, default="PNGImages_01", help="Base image directory")
    parser.add_argument("--compute_flow", action="store_true", help="Flag to compute optical flow")
    parser.add_argument("--optical_threshold", type=float, default=0.5, help="Optical flow threshold for filtering data")
    parser.add_argument("--corner_size", type=float, default=0.15, help="Size of the corner for optical flow processing")
    parser.add_argument("--sequence_length", type=int, default=5, help="Minimal number of consecutive frames")
    parser.add_argument("--required_corners", type=int, default=3, help="Minimal number of corners to meet the optical flow threshold requirement")
    parser.add_argument("--model_path",type=str,default='raft/checkpoints/raft-kitti.pth',help="Path of pretrained raft weights")
    parser.add_argument("--reverse_flag",type=bool,default=False,help="Compute reverse flow")
    parser.add_argument("--diod_pd", action="store_true", help="Flag to perform additional image processing (resize and crop)")
    parser.add_argument("--pixel_threshold", type=float, default=2.5, help="Optical flow threshold to convert csv to png")
    args = parser.parse_args()
    main(args)
