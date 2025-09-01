import os
import argparse
from raft import run_inference
from utils import subset_selection_of as sof, data, binary_map2_instances, kitti_samples_removing

def create_output_directories(*directories):
    """Create multiple directories from a list."""
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def main(args):
    base_input_dir = os.path.join(args.base_dir, args.base_input_dir)
    Flows_gap1 = os.path.join(base_input_dir, "Flows_gap1")
    FlowsImages_gap1 = os.path.join(base_input_dir, "FlowImages_gap1")
    Flows_value_gap1 = os.path.join(base_input_dir, "Flows_value_gap1")
    base_of_output_dir = os.path.join(
        args.base_dir,
        f"KITTI_{args.optical_threshold}_{args.rgb_path}",
    )
    if args.compute_flow:
        run_inference.process_images(
            base_input_dir, args.rgb_path, args.model_path, args.reverse_flag, args.image_res
        )
        data.process_and_save_corner_flows(Flows_gap1, Flows_value_gap1, args.corner_size)

    sof.filter_and_save_images_by_corner_flow(
        base_input_dir,
        args.rgb_path,
        base_of_output_dir,
        "Flows_value_gap1",
        args.optical_threshold,
        args.sequence_length,
        args.required_corners,
    )
    sof.sync_optical_flow_images(
        FlowsImages_gap1,
        os.path.join(base_of_output_dir, "optical_flow"),
        os.path.join(base_of_output_dir, "optical_flow_image"),
    )
    sof.sync_optical_flow(
        Flows_gap1,
        os.path.join(base_of_output_dir, "optical_flow"),
        os.path.join(base_of_output_dir, "Flows_gap1"),
    )
    data.recursive_csv_to_png(
        os.path.join(base_of_output_dir, "optical_flow"), base_of_output_dir, args.pixel_threshold
    )
    data.filter_and_delete_images(
        os.path.join(base_of_output_dir, f"init_seg_{args.pixel_threshold}")
    )
    binary_map2_instances.process_folder_files(base_of_output_dir, args.pixel_threshold)
    data.batch_merge_pngs(
        os.path.join(base_of_output_dir, f"instances_{args.pixel_threshold}"),
        os.path.join(base_of_output_dir, f"init_seg_{args.pixel_threshold}"),
        os.path.join(base_of_output_dir, "optical_flow_image"),
        os.path.join(base_of_output_dir, args.rgb_path),
        os.path.join(base_of_output_dir, f"combination_{args.pixel_threshold}"),
    )
    if args.rgb_path=="PNGImages_02":
        kitti_samples_removing.remove_kitti_files(base_of_output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process KITTI Dataset.")
    parser.add_argument("--base_dir",type=str,
        default="../scripts/datasets/data/KITTI",
        help="Base directory for processing and output",
    )
    parser.add_argument(
        "--base_input_dir", type=str, default="KITTI_train/image_02", help="Base input directory"
    )
    parser.add_argument("--rgb_path", type=str, default="PNGImages_02", help="Base image directory")
    parser.add_argument("--compute_flow", action="store_true", help="Flag to compute optical flow")
    parser.add_argument("--optical_threshold", type=float, default=1.7, help="Optical flow threshold for filtering data")
    parser.add_argument("--corner_size", type=float, default=0.15, help="Size of the corner for optical flow processing")
    parser.add_argument("--sequence_length", type=int, default=5, help="Minimal number of consecutive frames")
    parser.add_argument("--required_corners", type=int, default=3, help="Minimal number of corners to meet the optical flow threshold requirement")
    parser.add_argument("--model_path", type=str, default="raft/checkpoints/raft-kitti.pth", help="Path of pretrained raft weights")
    parser.add_argument("--reverse_flag", type=bool, default=False, help="Compute reverse flow")
    parser.add_argument("--image_res", type=tuple, default=(1240, 376), help="Resize the images to the specified resolution. Keep original resolution: (0,0)")
    parser.add_argument("--pixel_threshold", type=float, default=2.5, help="Optical flow threshold to convert csv to png")
    args = parser.parse_args()
    main(args)