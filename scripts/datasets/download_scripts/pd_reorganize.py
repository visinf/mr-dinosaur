# ######################################################## Process Training Set of PD#########################################################################
# (1) Delete scene folders listed in `banned_scenes`.
# (2) Traverse and clean the directory structure, keeping only /rgb/camera_01, /rgb/camera_05, /rgb/camera_06 in each scene.
# (3) Reorganize the dataset so that images of each scene are placed under top-level camera folders:
#     New structure: /camera_X/JPEGImages_XX/scene_XXXXXX/xxxx.png

banned_scenes = [
    "scene_000100",
    "scene_000002",
    "scene_000008",
    "scene_000012",
    "scene_000018",
    "scene_000029",
    "scene_000038",
    "scene_000040",
    "scene_000043",
    "scene_000044",
    "scene_000049",
    "scene_000050",
    "scene_000053",
    "scene_000063",
    "scene_000079",
    "scene_000090",
    "scene_000094",
    "scene_000100",
    "scene_000103",
    "scene_000106",
    "scene_000111",
    "scene_000112",
    "scene_000124",
    "scene_000125",
    "scene_000127",
    "scene_000148",
    "scene_000159",
    "scene_000166",
    "scene_000169",
    "scene_000170",
    "scene_000171",
    "scene_000187",
    "scene_000191",
    "scene_000200",
    "scene_000202",
    "scene_000217",
    "scene_000218",
    "scene_000225",
    "scene_000229",
    "scene_000232",
    "scene_000236",
    "scene_000237",
    "scene_000245",
    "scene_000249",
]


def delete_banned_scenes(top_dir, banned_scenes):
    """
    Delete scene folders that are listed in `banned_scenes`.
    """

    for root, dirs, files in os.walk(top_dir):
        for dir_name in dirs:
            if dir_name in banned_scenes:
                dir_path = os.path.join(root, dir_name)
                print(f"Deleting: {dir_path}")
                shutil.rmtree(dir_path)


def clean_scenes_directory(base_path):
    """
    Traverse and clean the given directory structure, keeping only /rgb/camera_* folders
    (specifically camera_01, camera_05, camera_06) within each scene folder.
    """
    if not os.path.exists(base_path):
        print(f"Path Not Found: {base_path}")
        return

    for scene_dir in os.listdir(base_path):
        scene_path = os.path.join(base_path, scene_dir)
        if os.path.isdir(scene_path):
            rgb_path = os.path.join(scene_path, "rgb")

            if not os.path.exists(rgb_path):
                continue

            for folder in os.listdir(scene_path):
                folder_path = os.path.join(scene_path, folder)
                if os.path.isdir(folder_path) and folder not in ["rgb"]:
                    shutil.rmtree(folder_path)
                    print(f"Deleting: {folder_path}")

            for folder in os.listdir(rgb_path):
                folder_path = os.path.join(rgb_path, folder)
                if os.path.isdir(folder_path) and folder not in [
                    "camera_01",
                    "camera_05",
                    "camera_06",
                ]:
                    shutil.rmtree(folder_path)
                    print(f"Deleting: {folder_path}")


def reorganize_train_dataset(base_path):
    """
    Reorganize the dataset so that images of each scene are placed under top-level camera folders:
    /camera_X/JPEGImages_XX/scene_XXXXXX/xxxx.png
    """
    if not os.path.exists(base_path):
        print(f"Path Not Found: {base_path}")
        return

    camera_folders = {
        "camera_01": "JPEGImages_01",
        "camera_05": "JPEGImages_05",
        "camera_06": "JPEGImages_06",
    }

    for scene_dir in os.listdir(base_path):
        scene_path = os.path.join(base_path, scene_dir)

        if os.path.isdir(scene_path):
            rgb_path = os.path.join(scene_path, "rgb")

            if os.path.exists(rgb_path):
                for camera, jpeg_folder in camera_folders.items():
                    camera_path = os.path.join(rgb_path, camera)

                    if os.path.exists(camera_path):
                        new_camera_dir = os.path.join(
                            base_path, camera, jpeg_folder, scene_dir
                        )
                        os.makedirs(new_camera_dir, exist_ok=True)

                        for image_file in os.listdir(camera_path):
                            src_image_path = os.path.join(camera_path, image_file)
                            if os.path.isfile(src_image_path):
                                dst_image_path = os.path.join(
                                    new_camera_dir, image_file
                                )
                                shutil.move(src_image_path, dst_image_path)
                                print(f"Move: {src_image_path} to {dst_image_path}")

                        shutil.rmtree(camera_path)
                        print(f"Deleting: {camera_path}")
                    else:
                        print(f"Folder Not Found: {camera_path}")

                shutil.rmtree(rgb_path)
                print(f"Deleting: {rgb_path}")

            shutil.rmtree(scene_path)
            print(f"Deleting: {scene_path}")

    print("Completed")
################################################ Process Validation Set of PD ##################################################################################

import os
import shutil


def reorganize_dataset(base_path):
    """
    Reorganize the PD test video dataset by moving the contents under each
    scene folder into a new folder hierarchy.

    Args:
        base_path (str): Root directory path of the dataset.
    """

    target_rgb_path = os.path.join(base_path, "rgb")
    target_ari_masks_path = os.path.join(base_path, "ari_masks")

    os.makedirs(target_rgb_path, exist_ok=True)
    os.makedirs(target_ari_masks_path, exist_ok=True)

    for scene_folder in os.listdir(base_path):
        scene_path = os.path.join(base_path, scene_folder)

        if os.path.isdir(scene_path) and scene_folder.startswith("scene_"):

            for folder_name in ["ari_masks", "rgb"]:
                source_folder_path = os.path.join(scene_path, folder_name)
                if os.path.isdir(source_folder_path):
                    target_folder_path = os.path.join(
                        base_path, folder_name, scene_folder
                    )

                    for camera_folder in os.listdir(source_folder_path):
                        camera_folder_path = os.path.join(
                            source_folder_path, camera_folder
                        )

                        if os.path.isdir(camera_folder_path):
                            target_camera_folder_path = os.path.join(
                                target_folder_path, camera_folder
                            )

                            os.makedirs(target_camera_folder_path, exist_ok=True)

                            for image_file in os.listdir(camera_folder_path):
                                source_image_path = os.path.join(
                                    camera_folder_path, image_file
                                )
                                target_image_path = os.path.join(
                                    target_camera_folder_path, image_file
                                )
                                shutil.move(source_image_path, target_image_path)

    for scene_folder in os.listdir(base_path):
        scene_path = os.path.join(base_path, scene_folder)
        if os.path.isdir(scene_path) and scene_folder.startswith("scene_"):
            shutil.rmtree(scene_path)



def move_and_rename_images_in_folder(folder_path):
    """
    Move all PNG files found in every scene and camera subfolder to the top‑level
    folder, renaming them sequentially.

    Args:
        folder_path (str): Path to the folder to process (e.g., ari_masks or rgb).
    """
    global_counter = 0

    for scene_folder in sorted(os.listdir(folder_path)):
        scene_path = os.path.join(folder_path, scene_folder)

        if os.path.isdir(scene_path):
            for camera_folder in sorted(os.listdir(scene_path)):
                camera_path = os.path.join(scene_path, camera_folder)

                if os.path.isdir(camera_path):
                    images = sorted(
                        [img for img in os.listdir(camera_path) if img.endswith(".png")]
                    )

                    for image_file in images:
                        new_name = f"{global_counter:010}.png"
                        source_image_path = os.path.join(camera_path, image_file)
                        target_image_path = os.path.join(folder_path, new_name)
                        shutil.move(source_image_path, target_image_path)
                        global_counter += 1

    for scene_folder in sorted(os.listdir(folder_path)):
        scene_path = os.path.join(folder_path, scene_folder)
        if os.path.isdir(scene_path):
            shutil.rmtree(scene_path)



def flatten_and_rename_ari_masks_and_rgb(base_path):
    """
    Flatten the PNG files inside the ari_masks and rgb folders by removing the
    intermediate scene and camera directories, and rename them sequentially.

    Args:
        base_path (str): Root directory path of the dataset.
    """

    ari_masks_path = os.path.join(base_path, "ari_masks")
    move_and_rename_images_in_folder(ari_masks_path)

    rgb_path = os.path.join(base_path, "rgb")
    move_and_rename_images_in_folder(rgb_path)
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Reorganize PD dataset for training or validation."
    )
    parser.add_argument(
        "--mode",
        choices=["train", "val"],
        default="train",
        help=(
            "Select processing mode: 'train' will delete banned scenes and keep "
            "only the required cameras; 'val' only reorganizes the structure."
        ),
    )
    parser.add_argument(
        "--base_path",
        required=True,
        help=(
            "Root directory of the dataset"
        ),
    )

    args = parser.parse_args()
    base_path = args.base_path

    if args.mode == "train":
        delete_banned_scenes(base_path, banned_scenes)
        clean_scenes_directory(base_path)
        reorganize_train_dataset(base_path)
    else:
        reorganize_dataset(base_path)
        flatten_and_rename_ari_masks_and_rgb(base_path)
