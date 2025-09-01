import os
import shutil
import argparse
import re

def _natural_key(name: str):
    return [int(s) if s.isdigit() else s.lower()
            for s in re.split(r'(\d+)', name)]
def merge_folders(folders, output_folder):
    """
    Merge tar files from multiple folders into a new folder with unique file names,
    starting from shard-000000.tar.
    
    :param folders: List of folder paths containing tar files to merge.
    :param output_folder: Path to the output folder where merged files will be stored.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize the starting index for the new files
    file_index = 0
    
    # Iterate through each folder
    for folder in folders:
        folder_files = sorted(
            (f for f in os.listdir(folder) if f.endswith('.tar')),
            key=_natural_key
        )
        print(folder_files)
        # Move and rename files
        for file in folder_files:
            new_filename = f"shard-{file_index:06}.tar"
            shutil.move(os.path.join(folder, file), os.path.join(output_folder, new_filename))
            print(f"移动文件: {os.path.join(folder, file)} 到 {os.path.join(output_folder, new_filename)}")
            file_index += 1
        
        # 删除原始文件夹
        shutil.rmtree(folder)
        print(f"删除文件夹：{folder}")

    print(f"文件已成功合并到 '{output_folder}'，文件名从 shard-000000.tar 开始依次递增。")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Merge tar files from multiple folders into a new folder with unique file names.')
    parser.add_argument('folders', type=str, nargs='+', help='Paths to the folders containing tar files.')
    parser.add_argument('output_folder', type=str, help='Path to the output folder where merged files will be stored.')
    
    # Parse arguments
    args = parser.parse_args()

    # Call the merge function with provided folder paths
    merge_folders(args.folders, args.output_folder)

if __name__ == '__main__':
    main()
