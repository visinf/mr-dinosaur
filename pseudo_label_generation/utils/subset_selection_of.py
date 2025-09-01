import os
import shutil
import json
from concurrent.futures import ThreadPoolExecutor

def process_folder(folder, optical_flow_path, png_image_path, new_optical_flow_path, new_png_image_path, threshold, sequence_length, required_corners):
    folder_path = os.path.join(optical_flow_path, folder)
    json_file_path = os.path.join(folder_path, f"{folder}.json")
    if os.path.isfile(json_file_path):
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)
            keys = list(data.keys())

            segment_index = -1
            i = 0

            while i < len(keys) - sequence_length + 1:
                if all(
                    sum(
                        data[keys[j]][corner] < threshold
                        for corner in ["top_left", "top_right", "bottom_left", "bottom_right"]
                    ) >= required_corners
                    for j in range(i, i + sequence_length)
                ):
                    segment_index += 1
                    segment_start = i

                    target_flow_folder = os.path.join(new_optical_flow_path, folder, f"{folder}_seg{segment_index}")
                    target_png_folder = os.path.join(new_png_image_path, folder, f"{folder}_seg{segment_index}")
                    os.makedirs(target_flow_folder, exist_ok=True)
                    os.makedirs(target_png_folder, exist_ok=True)

                    filtered_dict = {}

                    while (
                        i < len(keys)
                        and sum(
                            data[keys[i]][corner] < threshold
                            for corner in ["top_left", "top_right", "bottom_left", "bottom_right"]
                        ) >= required_corners
                    ):
                        key = keys[i]
                        filtered_dict[key] = data[key]

                        flow_img = key + ".csv"
                        png_img = key + ".png"
                        source_flow_img = os.path.join(folder_path, flow_img)
                        source_png_img = os.path.join(png_image_path, folder, png_img)

                        shutil.copy(source_flow_img, target_flow_folder)
                        shutil.copy(source_png_img, target_png_folder)
                        i += 1

                    if filtered_dict:
                        filtered_json_path = os.path.join(target_flow_folder, f"{folder}_seg{segment_index}_filtered.json")
                        with open(filtered_json_path, "w") as f:
                            json.dump(filtered_dict, f, indent=4)
                else:
                    i += 1

def filter_and_save_images_by_corner_flow(base_path, image_path, new_base_path, optical_flow_folder="Flows_value_gap1", threshold=1, sequence_length=4, required_corners=3, num_workers=8):
    optical_flow_path = os.path.join(base_path, optical_flow_folder)

    png_image_path = os.path.join(base_path, image_path)

    new_optical_flow_path = os.path.join(new_base_path, "optical_flow")
    new_png_image_path = os.path.join(new_base_path, image_path)

    os.makedirs(new_optical_flow_path, exist_ok=True)
    os.makedirs(new_png_image_path, exist_ok=True)

    folders = os.listdir(optical_flow_path)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        executor.map(
            lambda folder: process_folder(
                folder, optical_flow_path, png_image_path, new_optical_flow_path, new_png_image_path,
                threshold, sequence_length, required_corners
            ),
            folders
        )


def merge_all_json_files(base_dir):

    merged_data = {}

    # Traverse all subdirectories and files within base_dir
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".json"):
                json_file_path = os.path.join(root, file)

                with open(json_file_path, "r") as json_file:
                    data = json.load(json_file)

                    for key, value in data.items():
                        if key in merged_data:
                            merged_data[key].update(value)
                        else:
                            merged_data[key] = value

    # Write the merged data to a new JSON file in the optical_flow directory
    with open(
        os.path.join(base_dir, "optical_flow", "merged_data.json"), "w"
    ) as merged_file:
        json.dump(merged_data, merged_file, indent=4)


def load_keys_from_json_files(directory):

    keys = set()
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                json_file_path = os.path.join(root, file)
                with open(json_file_path, "r") as json_file:
                    data = json.load(json_file)
                    keys.update(data.keys())
    return keys

def sync_optical_flow_images(flow_images_base_dir, flow_csv_dir, flow_images_output_dir):
    for root, dirs, files in os.walk(flow_csv_dir):
        for file in files:
            if file.endswith('.csv'):
                png_filename = file.replace('.csv', '.png')

                seq_name = root.split(os.sep)[-2] 
                source_png_path = os.path.join(flow_images_base_dir, seq_name, png_filename)

                if os.path.exists(source_png_path):
                    target_dir_path = root.replace(flow_csv_dir, flow_images_output_dir)
                    os.makedirs(target_dir_path, exist_ok=True) 
                    target_png_path = os.path.join(target_dir_path, png_filename)
                    
                    shutil.copy2(source_png_path, target_png_path)
                    
def sync_optical_flow(flow_base_dir, flow_csv_dir, flow_output_dir):
    for root, dirs, files in os.walk(flow_csv_dir):
        for file in files:
            if file.endswith('.csv'):
                png_filename = file.replace('.csv', '.flo')

                seq_name = root.split(os.sep)[-2] 
                source_flo_path = os.path.join(flow_base_dir, seq_name, png_filename)

                if os.path.exists(source_flo_path):
                    target_dir_path = root.replace(flow_csv_dir, flow_output_dir)
                    os.makedirs(target_dir_path, exist_ok=True)  
                    target_flo_path = os.path.join(target_dir_path, png_filename)
                    
                    shutil.copy2(source_flo_path, target_flo_path)
  
def sync_optical_flow_smurf(flow_base_dir, flow_csv_dir, flow_output_dir):
    for root, dirs, files in os.walk(flow_csv_dir):
        for file in files:
            if file.endswith('.csv'):
                png_filename = file.replace('.csv', '.npy')

                seq_name = root.split(os.sep)[-2] 
                source_flo_path = os.path.join(flow_base_dir, seq_name, png_filename)

                if os.path.exists(source_flo_path):
                    target_dir_path = root.replace(flow_csv_dir, flow_output_dir)
                    os.makedirs(target_dir_path, exist_ok=True) 
                    target_flo_path = os.path.join(target_dir_path, png_filename)
                    
                    shutil.copy2(source_flo_path, target_flo_path)
