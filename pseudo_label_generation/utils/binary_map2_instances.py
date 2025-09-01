import numpy as np
import pandas as pd
from skimage import measure
from skimage.io import imread, imsave
from skimage import morphology
import os
import matplotlib
from scipy.ndimage import sobel
import hdbscan
from concurrent.futures import ThreadPoolExecutor
from functools import partial


def read_flo_file(filename):
    with open(filename, "rb") as f:
        magic = np.fromfile(f, np.float32, count=1)[0]
        if magic != 202021.25:
            raise ValueError("Magic number incorrect. Invalid .flo file")
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2 * w * h)
        return np.resize(data, (h, w, 2))


def calculate_internal_edges(flow_map, region_mask, erosion_size=10):
    eroded_mask = morphology.binary_erosion(region_mask, morphology.disk(erosion_size))

    gradient_x = sobel(flow_map[..., 0])
    gradient_y = sobel(flow_map[..., 1])

    edges = np.sqrt(gradient_x**2 + gradient_y**2)
    internal_edges = edges * eroded_mask

    return internal_edges


def encode_flow_direction(flow_map, num_bins=16):
    angles = np.arctan2(flow_map[..., 1], flow_map[..., 0])
    angles = (angles + 2 * np.pi) % (2 * np.pi)
    bin_size = 2 * np.pi / num_bins
    direction_encoding = np.floor(angles / bin_size).astype(int)
    return direction_encoding


def assign_instance_ids(
    binary_map,
    flow_csv,
    flow_map,
    edge_threshold=20,
    spatial_weight=60,
    direction_weight=100,
    min_region_size=100,
):
    labeled_map, num_features = measure.label(binary_map, return_num=True)  
    instance_map = np.zeros_like(labeled_map, dtype=np.int32)
    flow_direction_encoding = encode_flow_direction(flow_map)

    instance_id = 1
    for region_id in range(1, num_features + 1):

        region_mask = labeled_map == region_id
        if np.sum(region_mask) == 0:
            continue
        coords = np.column_stack(np.nonzero(region_mask))

        flow_region = flow_csv[region_mask]
        flow_direction = flow_direction_encoding[region_mask]
        flow_values = flow_map[region_mask]

        internal_edges = calculate_internal_edges(flow_map, region_mask)
        region_edge_strength = internal_edges[region_mask]

        if np.any(region_edge_strength > edge_threshold):
            features = np.hstack(
                (
                    flow_values,
                    coords * spatial_weight,
                    flow_direction.reshape(-1, 1) * direction_weight,
                )
            )
            hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=30, allow_single_cluster=False)
            labels = hdbscan_clusterer.fit_predict(features)

            unique_labels = np.unique(labels)
            label_mask = np.full(region_mask.shape, -1)
            label_mask[region_mask] = labels

            for label in unique_labels:
                if label == -1:

                    continue
                sub_region = label_mask == label
                instance_map[sub_region] = instance_id
                instance_id += 1

        else:
            instance_map[region_mask] = instance_id
            instance_id += 1
    unique_ids = np.unique(instance_map)
    for unique_id in unique_ids:
        if unique_id == 0:

            continue
        region_mask = instance_map == unique_id
        if np.sum(region_mask) < min_region_size:
            instance_map[region_mask] = 0
    return instance_map

def visualize_instance_map(instance_map):
    unique_ids = np.unique(instance_map)
    n_colors = len(unique_ids)
    cmap = matplotlib.colormaps.get_cmap("hsv")
    colors = cmap(np.linspace(0, 1, n_colors))

    colored_instance_map = np.zeros((*instance_map.shape, 3), dtype=np.uint8)
    for i, unique_id in enumerate(unique_ids):
        if unique_id == 0:
            continue
        mask = instance_map == unique_id
        colored_instance_map[mask] = (colors[i, :3] * 255).astype(np.uint8)

    return colored_instance_map


def process_specified_files(base_path, flow_threshold, index_dir):
    binary_map_path = os.path.join(base_path, f"init_seg_{flow_threshold}")
    csv_path = os.path.join(base_path, "optical_flow")
    flow_path = os.path.join(base_path, "Flows_gap1")
    index_file = os.path.join(base_path, index_dir)
    instance_dir_path = os.path.join(base_path, f"instances_{flow_threshold}")
    os.makedirs(instance_dir_path, exist_ok=True)

    with open(index_file, "r") as f:
        file_entries = f.readlines()

    for entry in file_entries:
        entry = entry.strip()
        sequence_name, timestamp = entry.split(";")

        # Traverse directories to find the matching files
        png_file = None
        csv_file = None
        flo_file = None

        for root, dirs, files in os.walk(binary_map_path):
            for file in files:
                if file == f"{sequence_name};{timestamp}.png":
                    png_file = os.path.join(root, file)
                    break

        for root, dirs, files in os.walk(csv_path):
            for file in files:
                if file == f"{sequence_name};{timestamp}.csv":
                    csv_file = os.path.join(root, file)
                    break

        for root, dirs, files in os.walk(flow_path):
            for file in files:
                if file == f"{sequence_name};{timestamp}.flo":
                    flo_file = os.path.join(root, file)
                    break

        if png_file and csv_file and flo_file:
            binary_map = imread(png_file, as_gray=True)
            flow_map = read_flo_file(flo_file)
            flow_csv = pd.read_csv(csv_file, header=None)
            flow_csv = flow_csv.to_numpy().reshape(binary_map.shape + (1,))

            # Assign instance IDs
            instance_map = assign_instance_ids(binary_map, flow_csv, flow_map)

            # Save the instance map and the visualized instance map
            instance_map_path = os.path.join(instance_dir_path, f"{sequence_name};{timestamp}.npy")
            colored_instance_map_path = os.path.join(
                instance_dir_path, f"{sequence_name};{timestamp}.png"
            )

            np.save(instance_map_path, instance_map)
            colored_instance_map = visualize_instance_map(instance_map)
            imsave(colored_instance_map_path, colored_instance_map)

def process_specified_files_for_instantiation(
    base_path, flow_threshold, index_dir, diff_threshold, edge_threshold
):

    binary_map_path = os.path.join(base_path, f"init_seg_{flow_threshold}")
    csv_path = os.path.join(base_path, "optical_flow")
    flow_path = os.path.join(base_path, "Flows_gap1")
    index_file = os.path.join(base_path, index_dir)
    instance_dir_path = os.path.join(
        base_path,
        f"instances_{flow_threshold}_diff_threshold_{diff_threshold}_edge_threshold{edge_threshold}",
    )
    os.makedirs(instance_dir_path, exist_ok=True)

    with open(index_file, "r") as f:
        file_entries = f.readlines()

    for entry in file_entries:
        entry = entry.strip()
        sequence_name, timestamp = entry.split(";")

        # Traverse directories to find the matching files
        png_file = None
        csv_file = None
        flo_file = None

        for root, dirs, files in os.walk(binary_map_path):
            for file in files:
                if file == f"{sequence_name};{timestamp}.png":
                    png_file = os.path.join(root, file)
                    break

        for root, dirs, files in os.walk(csv_path):
            for file in files:
                if file == f"{sequence_name};{timestamp}.csv":
                    csv_file = os.path.join(root, file)
                    break

        for root, dirs, files in os.walk(flow_path):
            for file in files:
                if file == f"{sequence_name};{timestamp}.flo":
                    flo_file = os.path.join(root, file)
                    break

        if png_file and csv_file and flo_file:
            binary_map = imread(png_file, as_gray=True)
            flow_map = read_flo_file(flo_file)
            flow_csv = pd.read_csv(csv_file, header=None)
            flow_csv = flow_csv.to_numpy().reshape(binary_map.shape + (1,))

            # Assign instance IDs
            instance_map = assign_instance_ids(
                binary_map, flow_csv, flow_map, diff_threshold, edge_threshold
            )

            # Save the instance map and the visualized instance map
            instance_map_path = os.path.join(instance_dir_path, f"{sequence_name};{timestamp}.npy")
            colored_instance_map_path = os.path.join(
                instance_dir_path, f"{sequence_name};{timestamp}.png"
            )

            np.save(instance_map_path, instance_map)
            colored_instance_map = visualize_instance_map(instance_map)
            imsave(colored_instance_map_path, colored_instance_map)


def process_specified_files_for_clustering(
    base_path, flow_threshold, index_dir, spatial_weight, direction_weight
):
    binary_map_path = os.path.join(base_path, f"init_seg_{flow_threshold}")
    csv_path = os.path.join(base_path, "optical_flow")
    flow_path = os.path.join(base_path, "Flows_gap1")
    index_file = os.path.join(base_path, index_dir)
    instance_dir_path = os.path.join(
        base_path,
        f"instances_{flow_threshold}_spatial_weight_{spatial_weight}_direction_weight{direction_weight}",
    )
    os.makedirs(instance_dir_path, exist_ok=True)

    with open(index_file, "r") as f:
        file_entries = f.readlines()

    for entry in file_entries:
        entry = entry.strip()
        sequence_name, timestamp = entry.split(";")

        # Traverse directories to find the matching files
        png_file = None
        csv_file = None
        flo_file = None

        for root, dirs, files in os.walk(binary_map_path):
            for file in files:
                if file == f"{sequence_name};{timestamp}.png":
                    png_file = os.path.join(root, file)
                    break

        for root, dirs, files in os.walk(csv_path):
            for file in files:
                if file == f"{sequence_name};{timestamp}.csv":
                    csv_file = os.path.join(root, file)
                    break

        for root, dirs, files in os.walk(flow_path):
            for file in files:
                if file == f"{sequence_name};{timestamp}.flo":
                    flo_file = os.path.join(root, file)
                    break

        if png_file and csv_file and flo_file:
            binary_map = imread(png_file, as_gray=True)
            flow_map = read_flo_file(flo_file)
            flow_csv = pd.read_csv(csv_file, header=None)
            flow_csv = flow_csv.to_numpy().reshape(binary_map.shape + (1,))

            # Assign instance IDs
            instance_map = assign_instance_ids(
                binary_map,
                flow_csv,
                flow_map,
                spatial_weight=spatial_weight,
                direction_weight=direction_weight,
            )

            # Save the instance map and the visualized instance map
            instance_map_path = os.path.join(instance_dir_path, f"{sequence_name};{timestamp}.npy")
            colored_instance_map_path = os.path.join(
                instance_dir_path, f"{sequence_name};{timestamp}.png"
            )

            np.save(instance_map_path, instance_map)
            colored_instance_map = visualize_instance_map(instance_map)
            imsave(colored_instance_map_path, colored_instance_map)


def process_file(file, root, binary_map_path, csv_path, flow_path, instance_dir_path):
    if file.endswith(".png"):
        # Get relative path from binary_map_path to the current file
        relative_path = os.path.relpath(root, binary_map_path)
        png_file_path = os.path.join(root, file)

        # Construct corresponding paths for CSV and FLO files
        csv_file_path = os.path.join(csv_path, relative_path, file.replace(".png", ".csv"))
        flo_file_path = os.path.join(flow_path, relative_path, file.replace(".png", ".flo"))

        # Check if all corresponding files exist
        if os.path.exists(csv_file_path) and os.path.exists(flo_file_path):
            binary_map = imread(png_file_path, as_gray=True)
            flow_map = read_flo_file(flo_file_path)  
            flow_csv = (
                pd.read_csv(csv_file_path, header=None).to_numpy().reshape(binary_map.shape + (1,))
            )

            # Assign instance IDs
            instance_map = assign_instance_ids(binary_map, flow_csv, flow_map)

            # Save the instance map and the visualized instance map
            instance_map_save_path = os.path.join(instance_dir_path, relative_path)
            os.makedirs(instance_map_save_path, exist_ok=True)

            instance_map_file = os.path.join(instance_map_save_path, file.replace(".png", ".npy"))
            colored_instance_map_file = os.path.join(instance_map_save_path, file)

            np.save(instance_map_file, instance_map)
            colored_instance_map = visualize_instance_map(instance_map)
            imsave(colored_instance_map_file, colored_instance_map)


def process_folder_files(base_path, flow_threshold, num_workers=8):
    # Load paths
    binary_map_path = os.path.join(base_path, f"init_seg_{flow_threshold}")
    csv_path = os.path.join(base_path, "optical_flow")
    flow_path = os.path.join(base_path, "Flows_gap1")
    instance_dir_path = os.path.join(base_path, f"instances_{flow_threshold}")
    os.makedirs(instance_dir_path, exist_ok=True)

    # List of tasks to process
    tasks = []

    # Traverse binary_map_path to find and process files
    for root, dirs, files in os.walk(binary_map_path):
        for file in files:
            tasks.append((file, root))

    # Define partial function with common arguments
    process_file_partial = partial(
        process_file,
        binary_map_path=binary_map_path,
        csv_path=csv_path,
        flow_path=flow_path,
        instance_dir_path=instance_dir_path,
    )

    # Use ThreadPoolExecutor to process files in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        executor.map(lambda args: process_file_partial(*args), tasks)


def process_file_smurf(file, root, binary_map_path, csv_path, flow_path, instance_dir_path):

    if file.endswith(".png"):
        # Get relative path from binary_map_path to the current file
        relative_path = os.path.relpath(root, binary_map_path)
        png_file_path = os.path.join(root, file)

        # Construct corresponding paths for CSV and FLO files
        csv_file_path = os.path.join(csv_path, relative_path, file.replace(".png", ".csv"))
        flo_file_path = os.path.join(flow_path, relative_path, file.replace(".png", ".npy"))

        # Check if all corresponding files exist
        if os.path.exists(csv_file_path) and os.path.exists(flo_file_path):
            binary_map = imread(png_file_path, as_gray=True)
            flow_map = np.load(flo_file_path)  
            flow_csv = (
                pd.read_csv(csv_file_path, header=None).to_numpy().reshape(binary_map.shape + (1,))
            )

            # Assign instance IDs
            instance_map = assign_instance_ids(binary_map, flow_csv, flow_map)

            # Save the instance map and the visualized instance map
            instance_map_save_path = os.path.join(instance_dir_path, relative_path)
            os.makedirs(instance_map_save_path, exist_ok=True)

            instance_map_file = os.path.join(instance_map_save_path, file.replace(".png", ".npy"))
            colored_instance_map_file = os.path.join(instance_map_save_path, file)
            np.save(instance_map_file, instance_map)
            colored_instance_map = visualize_instance_map(instance_map)
            imsave(colored_instance_map_file, colored_instance_map)
        else:
            print("wrong")


def process_folder_files_smurf(base_path, flow_threshold, num_workers=1):
    # Load paths
    binary_map_path = os.path.join(base_path, f"init_seg_{flow_threshold}")
    csv_path = os.path.join(base_path, "optical_flow")
    flow_path = os.path.join(base_path, "Flows_gap1_smurf")
    instance_dir_path = os.path.join(base_path, f"instances_{flow_threshold}")
    os.makedirs(instance_dir_path, exist_ok=True)
    # List of tasks to process
    tasks = []
    # Traverse binary_map_path to find and process files
    for root, dirs, files in os.walk(binary_map_path):
        for file in files:
            tasks.append((file, root))
    # Define partial function with common arguments
    process_file_partial = partial(
        process_file_smurf,
        binary_map_path=binary_map_path,
        csv_path=csv_path,
        flow_path=flow_path,
        instance_dir_path=instance_dir_path,
    )

    # Use ThreadPoolExecutor to process files in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        executor.map(lambda args: process_file_partial(*args), tasks)


