import logging
import logging
import os
import subprocess
from functools import lru_cache
from pathlib import Path
from PIL import Image
import cv2
import einops
import numpy as np
from cvbase.optflow.visualize import flow2rgb
import numpy as np
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor


__LOGGER = logging.Logger(__name__)
__TAR_SP = [Path("/usr/bin/tar"), Path("/bin/tar")]

TAG_FLOAT = 202021.25


@lru_cache(None)
def __tarbin():
    for p in __TAR_SP:
        if p.exists():
            return str(p)
    __LOGGER.error(f"Could not locate tar binary")
    return "tar"


def tar(*args):
    arg_list = [__tarbin(), *args]
    __LOGGER.info(f"Executing {arg_list}")
    print(f"Executing {arg_list}")
    return subprocess.check_call(arg_list, close_fds=True)


def read_flo(file):

    assert type(file) is str, "File parameter is not a string type: %r" % str(file)
    assert os.path.isfile(file) is True, "File not exists: %r" % str(file)
    assert file[-4:] == ".flo", "File extension is not '.flo': %r" % file[-4:]

    f = open(file, "rb")
    flo_number = np.fromfile(f, np.float32, count=1)[0]

    assert flo_number == TAG_FLOAT, "Flow number %r incorrect. Invalid .flo file" % flo_number

    w = np.fromfile(f, np.int32, count=1)
    h = np.fromfile(f, np.int32, count=1)
    data = np.fromfile(f, np.float32, count=2 * w[0] * h[0])
    flow = np.resize(data, (int(h), int(w), 2))

    f.close()
    return flow


def read_flo2(file):

    if not os.path.isfile(file):
        raise UnindetifedFlowError("File not exists : %r" % str(file))
    if str(file)[-4:] != ".flo":
        raise UnindetifedFlowError("File extension is not '.flo': %r" % file[-4:])

    try:
        with open(file, "rb") as f:
            flo_number = np.fromfile(f, np.float32, count=1)[0]
            if flo_number != TAG_FLOAT:
                raise UnindetifedFlowError("Flow tag number %r is incorrect. Invalid .flo file" % flo_number)
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            data = np.fromfile(f, np.float32, count=2 * w[0] * h[0])
            flow = np.resize(data, (int(h), int(w), 2))

    except (EOFError, IOError, OSError) as e:
        raise UnindetifedFlowError(str(e))

    return flow


def read_flow(sample_dir, resolution=(0, 0), to_rgb=False, ccrop=False, crop_frac=1.0):

    flow = read_flo(sample_dir)
    h, w, _ = np.shape(flow)

    if ccrop:
        s = int(min(h, w) * crop_frac)
        flow = flow[(h - s) // 2 : (h - s) // 2 + s, (w - s) // 2 : (w - s) // 2 + s]
        h, w = s, s  

    if resolution != (0, 0):
        flow = cv2.resize(flow, (resolution[0], resolution[1]), interpolation=cv2.INTER_NEAREST)
        flow[:, :, 0] *= resolution[0] / w
        flow[:, :, 1] *= resolution[1] / h

    if to_rgb:
        flow = np.clip((flow2rgb(flow) - 0.5) * 2, -1.0, 1.0)

    return einops.rearrange(flow, "h w c -> c h w")


def read_flow2(sample_dir, resolution=(0, 0), to_rgb=False, ccrop=False, crop_frac=1.0):

    flow = read_flo2(sample_dir)
    h, w, _ = np.shape(flow)

    if ccrop:
        s = int(min(h, w) * crop_frac)
        flow = flow[(h - s) // 2 : (h - s) // 2 + s, (w - s) // 2 : (w - s) // 2 + s]
        h, w = s, s

    if resolution != (0, 0):
        flow = cv2.resize(flow, (resolution[1], resolution[0]), interpolation=cv2.INTER_NEAREST)
        flow[:, :, 0] *= resolution[1] / w
        flow[:, :, 1] *= resolution[0] / h

    if to_rgb:
        flow = np.clip((flow2rgb(flow) - 0.5) * 2, -1.0, 1.0)

    return einops.rearrange(flow, "h w c -> c h w")


def process_and_save_corner_flows(
    base_dir, output_dir, corner_size=0.15, resolution=(0, 0), num_workers=8
):

    def process_flo_file(flo_file_path, output_subdir, corner_size):
        flow = read_flow(flo_file_path)
        _, h, w = flow.shape

        magnitude = np.sqrt(flow[0, :, :] ** 2 + flow[1, :, :] ** 2)

        corner_height = int(h * corner_size)
        corner_width = int(w * corner_size)

        corners = {
            "top_left": magnitude[:corner_height, :corner_width],
            "top_right": magnitude[:corner_height, -corner_width:],
            "bottom_left": magnitude[-corner_height:, :corner_width],
            "bottom_right": magnitude[-corner_height:, -corner_width:],
        }
        corner_averages = {corner: float(np.mean(corners[corner])) for corner in corners}

        output_file_path = os.path.join(
            output_subdir, os.path.basename(flo_file_path).replace(".flo", ".csv")
        )
        df = pd.DataFrame(magnitude)
        df.to_csv(output_file_path, index=False, header=False, float_format="%.3f")

        return os.path.basename(flo_file_path).replace(".flo", ""), corner_averages

    os.makedirs(output_dir, exist_ok=True)

    for subdir, dirs, files in os.walk(base_dir):
        corner_values = {}

        output_subdir = os.path.join(output_dir, os.path.relpath(subdir, base_dir))
        os.makedirs(output_subdir, exist_ok=True)

        flo_files = [f for f in files if f.endswith(".flo")]

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for file in flo_files:
                flo_file_path = os.path.join(subdir, file)
                futures.append(
                    executor.submit(process_flo_file, flo_file_path, output_subdir, corner_size)
                )

            for future in futures:
                file_name, corner_averages = future.result()
                corner_values[file_name] = corner_averages

        sorted_keys = sorted(corner_values.keys())
        sorted_corner_values = {key: corner_values[key] for key in sorted_keys}
        json_file_path = os.path.join(output_subdir, os.path.basename(subdir) + ".json")
        with open(json_file_path, "w") as json_file:
            json.dump(sorted_corner_values, json_file, indent=4)


def merge_json_files(base_output_folder):

    merged_data = {}

    # Traverse through the base output folder and its subdirectories
    for root, dirs, files in os.walk(base_output_folder):
        for file in files:
            # Check if the file is a JSON file
            if file.endswith(".json"):
                file_path = os.path.join(root, file)

                # Open and read the JSON file
                with open(file_path, "r") as json_file:
                    data = json.load(json_file)
                    # Merge the data
                    merged_data.update(data)

    # Save the merged data to a new JSON file
    output_file_path = os.path.join(base_output_folder, "merged_data.json")
    with open(output_file_path, "w") as output_file:
        json.dump(merged_data, output_file, indent=4)


def csv_to_png(csv_path, png_path, thresholding):
    data = pd.read_csv(csv_path, header=None)

    image_data = data.apply(lambda col: col.map(lambda x: 255 if x > thresholding else 0))
    image_array = image_data.to_numpy(dtype="uint8")
    img = Image.fromarray(image_array, "L")
    img.save(png_path)


def recursive_csv_to_png(source_dir, target_dir_base, thresholding, num_workers=8):
  
    target_dir = f"{target_dir_base}/init_seg_{thresholding}"
    os.makedirs(target_dir, exist_ok=True)

    def process_file(csv_file, target_subdir, filename):
        png_file = os.path.join(target_subdir, filename.replace(".csv", ".png"))
        data = pd.read_csv(csv_file, header=None)

        image_data = data.apply(lambda col: col.map(lambda x: 255 if x > thresholding else 0))
        image_array = image_data.to_numpy(dtype="uint8")
        img = Image.fromarray(image_array, "L")
        img.save(png_file)

    tasks = []
    idx = 0
    for root, _, files in os.walk(source_dir):
        for filename in files:
            if filename.endswith(".csv"):
                csv_file = os.path.join(root, filename)
                target_subdir = os.path.join(target_dir, os.path.relpath(root, source_dir))
                os.makedirs(target_subdir, exist_ok=True)

                tasks.append((csv_file, target_subdir, filename))
                idx += 1

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        executor.map(lambda args: process_file(*args), tasks)


def merge_pngs(paths, output_path):
    images = [Image.open(path) for path in paths]
    max_width = max(image.width for image in images)
    total_height = sum(image.height for image in images)

    new_img = Image.new("RGB", (max_width, total_height))
    y_offset = 0
    for image in images:
        x_offset = (max_width - image.width) // 2
        new_img.paste(image, (x_offset, y_offset))
        y_offset += image.height

    new_img.save(output_path)


def batch_merge_pngs(instance_dir, init_seg_dir, optical_flow_dir, png_dir, output_dir_base):
    print("Combining Images...")
    for root, _, files in os.walk(instance_dir):
        for filename in files:
            if filename.endswith(".png"):
                relative_path = os.path.relpath(root, instance_dir)
                paths = [
                    os.path.join(root, filename),
                    os.path.join(init_seg_dir, relative_path, filename),
                    os.path.join(optical_flow_dir, relative_path, filename),
                    os.path.join(png_dir, relative_path, filename),
                ]
                if all(os.path.exists(path) for path in paths):
                    output_subdir = os.path.join(output_dir_base, relative_path)
                    os.makedirs(output_subdir, exist_ok=True)
                    output_file_path = os.path.join(output_subdir, filename)
                    merge_pngs(paths, output_file_path)


def remove_empty_and_small_dirs(base_dir):
    for root, dirs, files in os.walk(base_dir, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            try:
                if os.path.exists(dir_path):
                    num_files = len(
                        [
                            f
                            for f in os.listdir(dir_path)
                            if os.path.isfile(os.path.join(dir_path, f))
                        ]
                    )
                    if num_files < 3:
                        for file in os.listdir(dir_path):
                            file_path = os.path.join(dir_path, file)
                            if os.path.isfile(file_path):
                                os.remove(file_path)
                        os.rmdir(dir_path)
                        print(f"Deleted directory with less than 3 files: {dir_path}")

            except OSError:
                pass

        if os.path.exists(root) and not os.listdir(root):
            os.rmdir(root)


def count_png_images(root_dir):
    png_count = 0

    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".png"):
                png_count += 1

    print(f"Total PNG images: {png_count}")

    return png_count


def filter_and_delete_images(init_seg_dir, pixel_value=255, threshold_ratio=0.0005):

    for root, _, files in os.walk(init_seg_dir):
        for filename in files:
            if filename.endswith(".png"):
                init_seg_path = os.path.join(root, filename)

                relative_path = os.path.relpath(init_seg_path, init_seg_dir)

                image = Image.open(init_seg_path).convert("L")
                image_np = np.array(image)
                total_pixels = image_np.size
                target_pixels = np.sum(image_np == pixel_value)

                if target_pixels < total_pixels * threshold_ratio:
                    os.remove(init_seg_path)
                    print(f"Deleted {relative_path}")

    for base_dir in [init_seg_dir]:
        remove_empty_and_small_dirs(base_dir)
    count_png_images(init_seg_dir)
