import argparse
import logging
import os
import random
from typing import Dict, Optional, List, Tuple
from PIL import Image
import numpy as np
import tqdm
import webdataset
from io import BytesIO

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
    """Reads raw bytes from file."""
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
    """Parse the training pseudo folder and return sorted list of image-label pairs."""
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
    return renamed_pairs


def resize_and_crop_image(
    img_path: str,
    new_size: Tuple[int, int],
    top_crop: int,
    is_label: bool = False,
    is_val_label: bool = False
) -> Image.Image:
    """
    打开图像/标签并做以下操作：
      1) 缩放到固定尺寸 new_size
      2) 从顶部裁剪掉 top_crop 像素
      3) 如果是验证集标签 (is_val_label=True)，去除像素数小于 500 的标签
         (像素数小于 500 的标签ID直接映射为 0，大于等于 500 的标签ID重新分配一个新的ID)
    """
    pil_img = Image.open(img_path)
    interp = Image.NEAREST if is_label else Image.BILINEAR
    pil_img = pil_img.resize(new_size, interp)
    left, upper, right, lower = 0, top_crop, new_size[0], new_size[1]
    pil_img = pil_img.crop((left, upper, right, lower))
    if is_val_label:
        mask = np.array(pil_img)
        values, indices, counts = np.unique(mask, return_inverse=True, return_counts=True)
        mapping = {}
        mapping_count = 0
        for i in range(len(values)):
            if counts[i] >= 500:
                mapping[values[i]] = mapping_count
                mapping_count += 1
            else:
                mapping[values[i]] = 0
        remapped = np.array([mapping[val] for val in values])[indices]
        remapped = remapped.reshape(mask.shape)
        pil_img = Image.fromarray(remapped.astype(np.uint8))

    return pil_img


def resize_and_crop_npy(
    npy_path: str,
    top_crop: int,
    new_size: Tuple[int, int]
) -> np.ndarray:
    """
    加载 .npy 格式的标签，使用最近邻插值将其缩放到 new_size，然后裁剪顶部 top_crop 像素。
    返回裁剪后的 numpy array。
    """
    label_array = np.load(npy_path)
    pil_img = Image.fromarray(label_array)
    pil_img = pil_img.resize(new_size, Image.NEAREST)
    left, upper, right, lower = 0, top_crop, new_size[0], new_size[1]
    pil_img = pil_img.crop((left, upper, right, lower))
    return np.array(pil_img)


def pil_to_bytes(pil_img: Image.Image, fmt: str = "PNG") -> bytes:
    buffer = BytesIO()
    pil_img.save(buffer, format=fmt)
    return buffer.getvalue()


def read_and_convert(label_array: np.ndarray, output_prefix: str):
    """
    生成 instance_mask.npy 和 instance_category.npy。
    """
    unique_ids, counts = np.unique(label_array, return_counts=True)
    mask_indices = np.where(unique_ids != 0)[0]
    instance_masks = []
    instance_categories = []
    
    background_mask = np.ones_like(label_array, dtype=np.uint8)
    for idx in mask_indices:
        unique_id = unique_ids[idx]
        binary_mask = (label_array == unique_id).astype(np.uint8)
        background_mask[binary_mask == 1] = 0
        instance_masks.append(binary_mask)
        instance_categories.append(unique_id)
    
    instance_masks.insert(0, background_mask)
    instance_categories.insert(0, 100)
    
    instance_masks = np.array(instance_masks)[..., None]
    instance_categories = np.array(instance_categories)
    instance_mask_path = f"{output_prefix}.instance_mask.npy"
    instance_category_path = f"{output_prefix}.instance_category.npy"

    np.save(instance_mask_path, instance_masks)
    np.save(instance_category_path, instance_categories)

    return instance_mask_path, instance_category_path

def additional_training_transform(
    pil_img: Image.Image,
    pil_lbl: Image.Image,
    output_prefix: str,
    new_size: Tuple[int, int],
    crop_size: Tuple[int, int]
) -> Tuple[Optional[Image.Image], Optional[Image.Image], Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    对已经经过初始 resize 和 crop 得到 (968,480) 的图像和标签，
    执行以下操作：
      1. 将图像使用 bilinear 插值、标签使用 nearest 插值分别 resize 到 (980,490)
      2. 将图像和标签从中间一分为二，分别保存两边（分别保存为PNG和Numpy格式）
      3. 如果某部分标签的唯一值小于等于1，则跳过该部分
    """

    # resize
    resized_img = pil_img.resize(new_size, Image.BILINEAR)
    resized_lbl = pil_lbl.resize(new_size, Image.NEAREST)

    left_img = resized_img.crop((0, 0, crop_size[0], crop_size[1]))
    right_img = resized_img.crop((crop_size[0], 0, new_size[0], new_size[1]))

    left_lbl = resized_lbl.crop((0, 0, crop_size[0], crop_size[1]))
    right_lbl = resized_lbl.crop((crop_size[0], 0, new_size[0], new_size[1]))

    left_lbl_array = np.array(left_lbl)
    right_lbl_array = np.array(right_lbl)

    left_valid = len(np.unique(left_lbl_array)) > 1
    right_valid = len(np.unique(right_lbl_array)) > 1
    if not left_valid and not right_valid:
        print(f"Skipping both parts due to labels with <=1 unique value.")
        return None, None, None, None, None, None  # Skip both parts if neither is valid
    prefix = os.path.basename(output_prefix)

    left_img_path = right_img_path = left_lbl_path = right_lbl_path = None
    left_key = right_key = None
    if left_valid:
        left_img_path = f"{output_prefix}_left.png"
        left_lbl_path = f"{output_prefix}_left.npy"
        left_key = f"{prefix}_left"
        left_img.save(left_img_path, format="PNG")
        np.save(left_lbl_path, np.array(left_lbl))

    if right_valid:
        right_img_path = f"{output_prefix}_right.png"
        right_lbl_path = f"{output_prefix}_right.npy"
        right_key = f"{prefix}_right"
        right_img.save(right_img_path, format="PNG")
        np.save(right_lbl_path, np.array(right_lbl))
    return left_img_path, right_img_path, left_lbl_path, right_lbl_path, left_key, right_key

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
    os.makedirs(output_path, exist_ok=True)

    from utils import FakeIndices, ContextList, get_shard_pattern, get_index
    patterns = [get_shard_pattern(output_path)]
    list_of_indices = [FakeIndices()]

    instance_count = 0
    with ContextList(webdataset.ShardWriter(p, **shard_writer_params) for p in patterns) as writers:
        for index, image_file in tqdm.tqdm(enumerate(image_files), total=dataset_length):
            instance = {
                '__key__': os.path.splitext(os.path.basename(image_file))[0]
            }
            if label_files:
                label_file = label_files[index]
                if not is_train:
                    pil_img = resize_and_crop_image(image_file, is_label=False, is_val_label=False,new_size=(968, 608), top_crop=128)
                    img_data = pil_to_bytes(pil_img, fmt="PNG")
                    pil_lbl = resize_and_crop_image(label_file, is_label=True, is_val_label=True,new_size=(968, 608), top_crop=128)
                    lbl_data = pil_to_bytes(pil_lbl, fmt="PNG")
                    output_prefix = os.path.join(output_path, instance['__key__'])
                    label_array = np.array(pil_lbl)
                    instance_mask_path, instance_category_path = read_and_convert(label_array, output_prefix)
                    instance['image.png'] = img_data
                    instance['label.png'] = lbl_data
                    instance['instance_mask.npy'] = read_file(instance_mask_path)
                    instance['instance_category.npy'] = read_file(instance_category_path)
                    writer_index = get_index(index, list_of_indices)
                    writer = writers[writer_index]
                    writer.write(instance)
                    os.remove(instance_mask_path)
                    os.remove(instance_category_path)
                else:
                    pil_img = Image.open(image_file)
                    lbl_array=np.load(label_file)
                    pil_lbl = Image.fromarray(lbl_array)

                    left_img_path, right_img_path, left_lbl_path, right_lbl_path, left_key, right_key = additional_training_transform(pil_img, pil_lbl, 
                                                                                                                                      os.path.join(output_path,instance['__key__']),
                                                                                                                                      new_size=(980, 490), 
                                                                                                                                      crop_size=(490, 490))
                    
                    # If the left part is valid, save it
                    if left_img_path is not None:
                        left_instance = instance.copy()  # Copy the original instance for the left part
                        left_instance['image.png'] = pil_to_bytes(Image.open(left_img_path), fmt="PNG")
                        left_instance['label.npy'] = read_file(left_lbl_path)
                        left_instance['__key__'] = left_key  # Assign unique key for the left part
                        os.remove(left_img_path)
                        os.remove(left_lbl_path)
                        writer_index = get_index(index, list_of_indices)
                        writer = writers[writer_index]
                        writer.write(left_instance)  # Write left part instance
                    # If the right part is valid, save it
                    if right_img_path is not None:
                        right_instance = instance.copy()  # Copy the original instance for the right part
                        right_instance['image.png'] = pil_to_bytes(Image.open(right_img_path), fmt="PNG")
                        right_instance['label.npy'] = read_file(right_lbl_path)
                        right_instance['__key__'] = right_key  # Assign unique key for the right part
                        os.remove(right_img_path)
                        os.remove(right_lbl_path)
                        writer_index = get_index(index, list_of_indices)
                        writer = writers[writer_index]
                        writer.write(right_instance)  # Write right part instance

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