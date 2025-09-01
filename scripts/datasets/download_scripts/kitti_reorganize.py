import os
import shutil
import argparse

def retain_and_reorganize(source_dir, folders_to_keep=["image_02", "image_03"]):
    """
    Traverse all date folders, retain only image_02 and image_03,
    and move PNGs to source_dir/image_xx/PNGImages_xx/scene_name/*.png
    Finally, remove empty scene and date folders.
    """
    for date_folder in os.listdir(source_dir):
        date_path = os.path.join(source_dir, date_folder)
        if not os.path.isdir(date_path):
            continue

        for scene_folder in os.listdir(date_path):
            scene_path = os.path.join(date_path, scene_folder)
            if not os.path.isdir(scene_path):
                continue

            for image_folder in os.listdir(scene_path):
                image_path = os.path.join(scene_path, image_folder)

                # Skip and delete folders not in keep list
                if image_folder not in folders_to_keep:
                    try:
                        shutil.rmtree(image_path)
                        print(f"Deleted: {image_path}")
                    except Exception as e:
                        print(f"Failed to delete {image_path}: {e}")
                    continue

                data_path = os.path.join(image_path, "data")
                if not os.path.isdir(data_path):
                    continue

                # Create global target directory
                global_target_dir = os.path.join(
                    source_dir,
                    image_folder,
                    f"PNGImages_{image_folder[-2:]}",
                    scene_folder
                )
                os.makedirs(global_target_dir, exist_ok=True)

                # Move all .png files
                for filename in os.listdir(data_path):
                    if filename.lower().endswith(".png"):
                        src_file = os.path.join(data_path, filename)
                        dst_file = os.path.join(global_target_dir, filename)
                        shutil.move(src_file, dst_file)

                # Delete 'data' folder
                try:
                    shutil.rmtree(data_path)
                    print(f"Moved and deleted: {data_path}")
                except Exception as e:
                    print(f"Failed to delete data folder {data_path}: {e}")

            # Remove empty scene folder
            if os.path.isdir(scene_path) and not os.listdir(scene_path):
                try:
                    shutil.rmtree(scene_path)
                    print(f"Removed empty scene folder: {scene_path}")
                except Exception as e:
                    print(f"Failed to remove scene folder {scene_path}: {e}")

        # Remove empty date folder
        if os.path.isdir(date_path):
            try:
                shutil.rmtree(date_path)
                print(f"Removed empty date folder: {date_path}")
            except Exception as e:
                print(f"Failed to remove date folder {date_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Reorganize KITTI dataset into global image_02/03 structure.")
    parser.add_argument('--source_dir', type=str, required=True, help="Root KITTI dataset directory")
    args = parser.parse_args()

    retain_and_reorganize(args.source_dir)

if __name__ == "__main__":
    main()
