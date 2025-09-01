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
from io import BytesIO

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

def parse_test_folder(image_dir: str) -> List[str]:
    """Parse the image folder and return sorted list of image files."""
    image_files = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.endswith('.png'):
                image_files.append(os.path.join(root, file))
    image_files.sort()
    return image_files

def parse_train_folder(image_dir: str) -> List[str]:

    image_files: List[str] = []

    for cam_folder in os.listdir(image_dir):
        cam_path = os.path.join(image_dir, cam_folder)
        if not os.path.isdir(cam_path):
            continue

        for sub in os.listdir(cam_path):
            if not sub.startswith('PNGImages'):
                continue

            png_dir = os.path.join(cam_path, sub)
            if not os.path.isdir(png_dir):
                continue

            for root, _, files in os.walk(png_dir):
                for file in files:
                    if file.endswith('.png'):
                        image_files.append(os.path.join(root, file))

    image_files.sort()
    return image_files

def parse_validation_folder(image_dir: str, label_dir: str) -> List[Tuple[str, str]]:
    """Parse the validation folder and return sorted list of image-label pairs."""
    image_files = []
    label_files = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.endswith('.png'):
                image_files.append(os.path.join(root, file))
    for root, _, files in os.walk(label_dir):
        for file in files:
            if file.endswith('.png'):
                label_files.append(os.path.join(root, file))
    image_files.sort()
    label_files.sort()
    if len(image_files) != len(label_files):
        raise ValueError("Mismatch between number of images and labels.")
    return list(zip(image_files, label_files))

def parse_train_pseudo_folder(image_dir: str, label_dir: str) -> List[Tuple[str, str]]:
    """
    Parse the training pseudo folder and return sorted list of image-label pairs.
    """
    image_files = []
    label_files = []
    
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.endswith('.png'):
                relative_path = os.path.relpath(os.path.join(root, file), image_dir)
                relative_path_no_ext = os.path.splitext(relative_path)[0]
                image_files.append((relative_path_no_ext, os.path.join(root, file)))
                
    for root, _, files in os.walk(label_dir):
        for file in files:
            # if file.endswith('.png'):
            if file.endswith('.npy'):
                relative_path = os.path.relpath(os.path.join(root, file), label_dir)
                relative_path_no_ext = os.path.splitext(relative_path)[0]
                label_files.append((relative_path_no_ext, os.path.join(root, file)))
    
    image_dict = {rel_path: full_path for rel_path, full_path in image_files}
    label_dict = {rel_path: full_path for rel_path, full_path in label_files}
    common_files = set(image_dict.keys()).intersection(set(label_dict.keys()))
    if not common_files:
        raise ValueError("No matching image-label pairs found.")
    
    paired_files = [(image_dict[file], label_dict[file]) for file in sorted(common_files)]
    renamed_pairs = []
    for i, (image_path, label_path) in enumerate(paired_files, 1):
        new_image_name = f"{i:05d}.png"  
        ext = os.path.splitext(label_path)[1]
        new_label_name = f"{i:05d}{ext}"  

        new_image_path = os.path.join(os.path.dirname(image_path), new_image_name)
        new_label_path = os.path.join(os.path.dirname(label_path), new_label_name)
        os.rename(image_path, new_image_path)
        os.rename(label_path, new_label_path)

        renamed_pairs.append((new_image_path, new_label_path))
    return renamed_pairs

def resize_and_split_image_label(image_path: str, label_path: str) -> Optional[List[Tuple[Image.Image, Image.Image, str]]]:
    """
    Resize image and label to (1260, 378), then split them into 4 overlapping windows.
    Each window is labeled with a unique key and invalid windows are skipped.
    """
    # Open image and label
    image = Image.open(image_path)
    if label_path.lower().endswith('.png'):
        label = Image.open(label_path)
    elif label_path.lower().endswith('.npy'):
        label_array = np.load(label_path)
        label = Image.fromarray(label_array)
    else:
        raise ValueError("Unsupported label file format: " + label_path)

    # Resize to (1260, 378)
    image = image.resize((1260, 378), Image.BILINEAR)
    label = label.resize((1260, 378), Image.NEAREST)

    # Define crop window sizes
    window_width, window_height = 378, 378
    crop_starts = [0, 294, 588, 882]  # Start indices for each crop window (with overlap)
    windows = []

    for i, start in enumerate(crop_starts):
        # Crop image and label
        image_cropped = image.crop((start, 0, start + window_width, window_height))
        label_cropped = label.crop((start, 0, start + window_width, window_height))

        # Check if label has any valid content (i.e., more than one unique value)
        label_array = np.array(label_cropped)
        if np.all(label_array == 0):  # Skip if label has no valid content
            continue

        # Generate unique key for each window
        key = f"{os.path.splitext(os.path.basename(image_path))[0]}_{i}"

        windows.append((image_cropped, label_cropped, key))

    return windows if windows else None



def read_and_convert(image_path, output_prefix):
    img_array = imageio.imread(image_path)
    unique_ids = np.unique(img_array)
    semantic_gt = unique_ids // 256
    instance_gt = unique_ids % 256
    mask_indices = np.where((semantic_gt >= 24) & (semantic_gt <= 33))[0]
    
    instance_masks = []
    instance_categories = []
    background_mask = np.ones_like(img_array, dtype=np.uint8)
    
    for idx in mask_indices:
        unique_id = unique_ids[idx]
        category = semantic_gt[idx]
        binary_mask = (img_array == unique_id).astype(np.uint8)
        background_mask[binary_mask == 1] = 0
        instance_masks.append(binary_mask)
        instance_categories.append(category)

    instance_masks.insert(0, background_mask)
    instance_categories.insert(0, 6)
    
    instance_masks = np.array(instance_masks)
    instance_masks = np.expand_dims(instance_masks, axis=-1)
    instance_categories = np.array(instance_categories)
    instance_mask_path = f"{output_prefix}.instance_mask.npy"
    instance_category_path = f"{output_prefix}.instance_category.npy"
    np.save(instance_mask_path, instance_masks)
    np.save(instance_category_path, instance_categories)

    return instance_mask_path, instance_category_path


def pil_to_bytes(pil_img: Image.Image, fmt: str = "PNG") -> bytes:
    buffer = BytesIO()
    pil_img.save(buffer, format=fmt)
    return buffer.getvalue()

def convert_to_webdataset(
    image_files: List[str],
    output_path: str,
    is_train: bool = True,
    label_files: Optional[List[str]] = None
):
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
            instance = {
                '__key__': os.path.splitext(os.path.basename(image_file))[0]
            }
            if not is_train and label_files:
                img_data=read_file(image_file)
                instance['image.png'] = img_data
                label_file = label_files[index]
                lbl_data = read_file(label_file)
                instance['label.png'] = lbl_data

                output_prefix = os.path.join(output_path, instance['__key__'])
                instance_mask_path, instance_category_path = read_and_convert(label_file, output_prefix)
                instance['instance_mask.npy'] = read_file(instance_mask_path)
                instance['instance_category.npy'] = read_file(instance_category_path)
                os.remove(instance_mask_path)
                os.remove(instance_category_path)
                writer_index = get_index(index, list_of_indices)
                writer = writers[writer_index]
                writer.write(instance)
                instance_count += 1
            elif is_train and label_files:
                label_file = label_files[index]
                split_results = resize_and_split_image_label(image_file, label_file)
                if split_results:
                    for image_cropped, label_cropped, key in split_results:
                        label_cropped=label_cropped.convert("L")
                        instance['image.png'] = pil_to_bytes(image_cropped, fmt="PNG")
                        label_array = np.array(label_cropped)
                        buf = BytesIO()
                        np.save(buf, label_array)
                        instance['label.npy'] = buf.getvalue()
                        instance['__key__'] = key  # Unique key for each split
                        writer_index = get_index(index, list_of_indices)
                        writer = writers[writer_index]
                        writer.write(instance)
                        instance_count += 1

                else:
                    logging.info(f"Skipping {image_file} due to invalid label.")
            else:
                img_data = read_file(image_file)
                instance['image.png'] = img_data

                writer_index = get_index(index, list_of_indices)
                writer = writers[writer_index]
                writer.write(instance)
                instance_count += 1

    logging.info(f"Wrote {instance_count} instances to {output_path}")

def main(
    train_image_dir: Optional[str],
    train_label_dir: Optional[str],
    val_image_dir: Optional[str],
    val_label_dir: Optional[str],
    test_image_dir: Optional[str],
    output_path: str,
    n_instances: Optional[int] = None,
    seed: int = 423234
):
    if train_image_dir:
        if not train_label_dir:
            train_image_files = parse_train_folder(train_image_dir)
            if n_instances:
                train_image_files = train_image_files[:n_instances]
            random.seed(seed)
            random.shuffle(train_image_files)
            convert_to_webdataset(train_image_files, output_path, is_train=True)
        else:
            train_image_label_pairs = parse_train_pseudo_folder(train_image_dir, train_label_dir)
            if n_instances:
                train_image_label_pairs = train_image_label_pairs[:n_instances]
            random.seed(seed)
            random.shuffle(train_image_label_pairs)
            image_files, label_files = zip(*train_image_label_pairs)
            convert_to_webdataset(list(image_files), output_path, is_train=True, label_files=list(label_files))
    elif val_image_dir and val_label_dir:
        val_image_label_pairs = parse_validation_folder(val_image_dir, val_label_dir)
        if n_instances:
            val_image_label_pairs = val_image_label_pairs[:n_instances]
        random.seed(seed)
        image_files, label_files = zip(*val_image_label_pairs)
        convert_to_webdataset(list(image_files), output_path, is_train=False, label_files=list(label_files))
    elif test_image_dir:
        test_image_files = parse_test_folder(test_image_dir)
        if n_instances:
            test_image_files = test_image_files[:n_instances]
        convert_to_webdataset(test_image_files, output_path, is_train=True)
    else:
        raise ValueError("No valid dataset directory provided.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_image_dir", type=str, help="Path to training images")
    parser.add_argument("--train_label_dir", type=str, help="Path to training labels")
    parser.add_argument("--val_image_dir", type=str, help="Path to validation images")
    parser.add_argument("--val_label_dir", type=str, help="Path to validation labels")
    parser.add_argument("--test_image_dir", type=str, help="Path to test images")
    parser.add_argument("output_path", type=str, help="Path to output webdataset")
    parser.add_argument("--n_instances", type=int, default=None, help="Number of instances to process")
    parser.add_argument("--seed", type=int, default=423234, help="Random seed")

    args = parser.parse_args()

    main(
        train_image_dir=args.train_image_dir,
        train_label_dir=args.train_label_dir,
        val_image_dir=args.val_image_dir,
        val_label_dir=args.val_label_dir,
        test_image_dir=args.test_image_dir,
        output_path=args.output_path,
        n_instances=args.n_instances,
        seed=args.seed,
    )
