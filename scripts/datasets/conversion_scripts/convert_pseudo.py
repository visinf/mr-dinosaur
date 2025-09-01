import argparse
import logging
import os
import random
from typing import Dict, Optional, List, Tuple
from PIL import Image
import numpy as np
import imageio
import tqdm
import webdataset
from utils import (
    ContextList,
    FakeIndices,
    create_split_indices,
    get_index,
    get_shard_pattern,
    make_subdirs_and_patterns,
)

logging.getLogger().setLevel(logging.INFO)

def determine_dataset_length(base_names, files):
    """Determine the length of a data folder dataset."""
    n_files = len(files)
    n_base_names = len(base_names)
    if not n_files / n_base_names == n_files // n_base_names:
        raise ValueError("Either some files are missing or non-related files in path.")
    return n_files // n_base_names

def read_file(path):
    with open(path, "rb") as f:
        return f.read()

def parse_train_pseudo_folder(image_dir: str, label_dir: str) -> List[Tuple[str, str]]:
    """Parse the validation folder and return sorted list of image-label pairs."""
    image_files = []
    label_files = []
    
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.endswith('.png'):
                # Include parent directory path and file name as key
                relative_path = os.path.relpath(os.path.join(root, file), image_dir)
                relative_path_no_ext = os.path.splitext(relative_path)[0]
                image_files.append((relative_path_no_ext, os.path.join(root, file)))
                
    for root, _, files in os.walk(label_dir):
        for file in files:
            if file.endswith('.npy'):
                # Include parent directory path and file name as key
                relative_path = os.path.relpath(os.path.join(root, file), label_dir)
                relative_path_no_ext = os.path.splitext(relative_path)[0]
                label_files.append((relative_path_no_ext, os.path.join(root, file)))

    # Create dictionaries with relative paths as keys
    image_dict = {rel_path: full_path for rel_path, full_path in image_files}
    label_dict = {rel_path: full_path for rel_path, full_path in label_files}
    # Find common relative paths
    common_files = set(image_dict.keys()).intersection(set(label_dict.keys()))
    if not common_files:
        raise ValueError("No matching image-label pairs found.")
    
    # Create sorted list of image-label pairs
    paired_files = [(image_dict[file], label_dict[file]) for file in sorted(common_files)]
    # Rename files
    renamed_pairs = []
    for i, (image_path, label_path) in enumerate(paired_files, 1):
        new_image_name = f"{i:05d}.png"  
        new_label_name = f"{i:05d}.npy"  

        new_image_path = os.path.join(os.path.dirname(image_path), new_image_name)
        new_label_path = os.path.join(os.path.dirname(label_path), new_label_name)

        os.rename(image_path, new_image_path)
        os.rename(label_path, new_label_path)

        renamed_pairs.append((new_image_path, new_label_path))
    return paired_files

def convert_to_webdataset(image_files: List[str], output_path: str, is_train: bool = True, label_files: Optional[List[str]] = None):
    shard_writer_params = {
        "maxsize": 100 * 1024 * 1024,  # 100 MB
        "maxcount": 1000,
        "keep_meta": True,
    }

    dataset_length = len(image_files)
    patterns = [get_shard_pattern(output_path)]
    os.makedirs(output_path, exist_ok=True)
    list_of_indices = [FakeIndices()]

    instance_count = 0
    with ContextList(webdataset.ShardWriter(p, **shard_writer_params) for p in patterns) as writers:
        for index, image_file in tqdm.tqdm(enumerate(image_files), total=dataset_length):
            img_data = read_file(image_file)

            instance = {
                'image.png': img_data,
                '__key__': os.path.splitext(os.path.basename(image_file))[0]
            }
            
            label_file = label_files[index]
            pil_lbl = np.load(label_file)
            if len(np.unique(pil_lbl)) <=1:
                print(f"Skipping {label_file} due to insufficient unique values.")
                continue
            lbl_data = read_file(label_file)
            instance['label.npy'] = lbl_data
            writer_index = get_index(index, list_of_indices)
            writer = writers[writer_index]
            writer.write(instance)
            instance_count += 1

    logging.info(f"Wrote {instance_count} instances to {output_path}")

def main(train_image_dir: Optional[str],train_label_dir: Optional[str], output_path: str, n_instances: Optional[int] = None, seed: int = 423234):
    if train_image_dir and train_label_dir:
        train_image_label_pairs = parse_train_pseudo_folder(train_image_dir, train_label_dir)
        if n_instances:
            train_image_label_pairs = train_image_label_pairs[:n_instances]
        random.seed(seed)
        random.shuffle(train_image_label_pairs)
        image_files, label_files = zip(*train_image_label_pairs)
        convert_to_webdataset(image_files, output_path, is_train=True, label_files=label_files)
    else:
        raise ValueError("No valid dataset directory provided, can only process training set with pseudo labels.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_image_dir", type=str, help="Path to training images")
    parser.add_argument("--train_label_dir", type=str, help="Path to training labels")
    parser.add_argument("output_path", type=str, help="Path to output webdataset")
    parser.add_argument("--n_instances", type=int, default=None, help="Number of instances to process")
    parser.add_argument("--seed", type=int, default=423234, help="Random seed")

    args = parser.parse_args()

    main(
        train_image_dir=args.train_image_dir,
        train_label_dir=args.train_label_dir,
        output_path=args.output_path,
        n_instances=args.n_instances,
        seed=args.seed,
    )
