import os
from typing import Tuple

KITTI_VAL_RAW: Tuple[str, ...] = (
    "2011_09_26/2011_09_26_drive_0015_sync/image_02/data/0000000264.png",
    "2011_09_26/2011_09_26_drive_0059_sync/image_02/data/0000000280.png",
    "2011_09_26/2011_09_26_drive_0014_sync/image_02/data/0000000020.png",
    "2011_09_26/2011_09_26_drive_0018_sync/image_02/data/0000000106.png",
    "2011_09_26/2011_09_26_drive_0014_sync/image_02/data/0000000192.png",
    "2011_09_26/2011_09_26_drive_0084_sync/image_02/data/0000000179.png",
    "2011_09_26/2011_09_26_drive_0028_sync/image_02/data/0000000228.png",
    "2011_09_29/2011_09_29_drive_0004_sync/image_02/data/0000000308.png",
    "2011_09_26/2011_09_26_drive_0013_sync/image_02/data/0000000020.png",
    "2011_09_26/2011_09_26_drive_0009_sync/image_02/data/0000000354.png",
    "2011_09_26/2011_09_26_drive_0056_sync/image_02/data/0000000122.png",
    "2011_09_26/2011_09_26_drive_0059_sync/image_02/data/0000000046.png",
    "2011_09_26/2011_09_26_drive_0032_sync/image_02/data/0000000218.png",
    "2011_09_29/2011_09_29_drive_0071_sync/image_02/data/0000000059.png",
    "2011_09_26/2011_09_26_drive_0028_sync/image_02/data/0000000313.png",
    "2011_09_26/2011_09_26_drive_0057_sync/image_02/data/0000000299.png",
    "2011_09_26/2011_09_26_drive_0011_sync/image_02/data/0000000147.png",
    "2011_09_26/2011_09_26_drive_0046_sync/image_02/data/0000000052.png",
    "2011_09_26/2011_09_26_drive_0051_sync/image_02/data/0000000218.png",
    "2011_09_26/2011_09_26_drive_0051_sync/image_02/data/0000000356.png",
    "2011_09_26/2011_09_26_drive_0051_sync/image_02/data/0000000322.png",
    "2011_10_03/2011_10_03_drive_0047_sync/image_02/data/0000000556.png",
    "2011_09_26/2011_09_26_drive_0014_sync/image_02/data/0000000240.png",
    "2011_09_26/2011_09_26_drive_0018_sync/image_02/data/0000000086.png",
    "2011_09_26/2011_09_26_drive_0104_sync/image_02/data/0000000035.png",
    "2011_09_26/2011_09_26_drive_0059_sync/image_02/data/0000000150.png",
    "2011_09_26/2011_09_26_drive_0013_sync/image_02/data/0000000070.png",
    "2011_09_26/2011_09_26_drive_0096_sync/image_02/data/0000000278.png",
    "2011_09_26/2011_09_26_drive_0015_sync/image_02/data/0000000286.png",
    "2011_09_26/2011_09_26_drive_0057_sync/image_02/data/0000000339.png",
    "2011_09_26/2011_09_26_drive_0011_sync/image_02/data/0000000167.png",
    "2011_09_26/2011_09_26_drive_0032_sync/image_02/data/0000000125.png",
    "2011_09_26/2011_09_26_drive_0056_sync/image_02/data/0000000191.png",
    "2011_09_26/2011_09_26_drive_0009_sync/image_02/data/0000000374.png",
    "2011_09_26/2011_09_26_drive_0032_sync/image_02/data/0000000340.png",
    "2011_09_26/2011_09_26_drive_0056_sync/image_02/data/0000000010.png",
    "2011_09_26/2011_09_26_drive_0028_sync/image_02/data/0000000284.png",
    "2011_09_26/2011_09_26_drive_0036_sync/image_02/data/0000000054.png",
    "2011_09_26/2011_09_26_drive_0051_sync/image_02/data/0000000282.png",
    "2011_09_26/2011_09_26_drive_0028_sync/image_02/data/0000000071.png",
    "2011_09_29/2011_09_29_drive_0004_sync/image_02/data/0000000258.png",
    "2011_09_26/2011_09_26_drive_0057_sync/image_02/data/0000000140.png",
    "2011_09_26/2011_09_26_drive_0011_sync/image_02/data/0000000111.png",
    "2011_09_26/2011_09_26_drive_0029_sync/image_02/data/0000000016.png",
    "2011_09_26/2011_09_26_drive_0059_sync/image_02/data/0000000300.png",
    "2011_09_29/2011_09_29_drive_0004_sync/image_02/data/0000000079.png",
    "2011_09_26/2011_09_26_drive_0014_sync/image_02/data/0000000050.png",
    "2011_09_26/2011_09_26_drive_0022_sync/image_02/data/0000000644.png",
    "2011_09_26/2011_09_26_drive_0017_sync/image_02/data/0000000050.png",
    "2011_09_26/2011_09_26_drive_0101_sync/image_02/data/0000000447.png",
    "2011_09_26/2011_09_26_drive_0015_sync/image_02/data/0000000219.png",
    "2011_09_26/2011_09_26_drive_0014_sync/image_02/data/0000000152.png",
    "2011_09_26/2011_09_26_drive_0070_sync/image_02/data/0000000224.png",
    "2011_09_26/2011_09_26_drive_0019_sync/image_02/data/0000000030.png",
    "2011_09_26/2011_09_26_drive_0032_sync/image_02/data/0000000095.png",
    "2011_09_26/2011_09_26_drive_0009_sync/image_02/data/0000000394.png",
    "2011_09_26/2011_09_26_drive_0056_sync/image_02/data/0000000282.png",
    "2011_09_26/2011_09_26_drive_0027_sync/image_02/data/0000000053.png",
    "2011_09_29/2011_09_29_drive_0004_sync/image_02/data/0000000105.png",
    "2011_09_26/2011_09_26_drive_0005_sync/image_02/data/0000000059.png",
    "2011_09_26/2011_09_26_drive_0019_sync/image_02/data/0000000097.png",
    "2011_09_26/2011_09_26_drive_0015_sync/image_02/data/0000000197.png",
    "2011_09_26/2011_09_26_drive_0018_sync/image_02/data/0000000066.png",
    "2011_09_26/2011_09_26_drive_0101_sync/image_02/data/0000000809.png",
    "2011_09_26/2011_09_26_drive_0015_sync/image_02/data/0000000239.png",
    "2011_09_26/2011_09_26_drive_0101_sync/image_02/data/0000000109.png",
    "2011_09_26/2011_09_26_drive_0017_sync/image_02/data/0000000030.png",
    "2011_09_26/2011_09_26_drive_0059_sync/image_02/data/0000000320.png",
    "2011_09_26/2011_09_26_drive_0014_sync/image_02/data/0000000129.png",
    "2011_09_26/2011_09_26_drive_0051_sync/image_02/data/0000000302.png",
    "2011_09_26/2011_09_26_drive_0032_sync/image_02/data/0000000378.png",
    "2011_09_26/2011_09_26_drive_0036_sync/image_02/data/0000000402.png",
    "2011_09_26/2011_09_26_drive_0051_sync/image_02/data/0000000342.png",
    "2011_09_26/2011_09_26_drive_0051_sync/image_02/data/0000000023.png",
    "2011_09_26/2011_09_26_drive_0056_sync/image_02/data/0000000082.png",
    "2011_09_26/2011_09_26_drive_0059_sync/image_02/data/0000000026.png",
    "2011_09_26/2011_09_26_drive_0013_sync/image_02/data/0000000010.png",
    "2011_09_26/2011_09_26_drive_0028_sync/image_02/data/0000000269.png",
    "2011_09_26/2011_09_26_drive_0011_sync/image_02/data/0000000127.png",
    "2011_09_26/2011_09_26_drive_0057_sync/image_02/data/0000000176.png",
    "2011_09_26/2011_09_26_drive_0028_sync/image_02/data/0000000378.png",
    "2011_09_26/2011_09_26_drive_0032_sync/image_02/data/0000000330.png",
    "2011_09_26/2011_09_26_drive_0014_sync/image_02/data/0000000010.png",
    "2011_09_26/2011_09_26_drive_0059_sync/image_02/data/0000000260.png",
    "2011_09_26/2011_09_26_drive_0015_sync/image_02/data/0000000273.png",
    "2011_09_29/2011_09_29_drive_0071_sync/image_02/data/0000000943.png",
    "2011_09_26/2011_09_26_drive_0096_sync/image_02/data/0000000020.png",
    "2011_09_26/2011_09_26_drive_0084_sync/image_02/data/0000000084.png",
    "2011_09_26/2011_09_26_drive_0014_sync/image_02/data/0000000172.png",
    "2011_09_26/2011_09_26_drive_0018_sync/image_02/data/0000000133.png",
    "2011_09_26/2011_09_26_drive_0009_sync/image_02/data/0000000364.png",
    "2011_09_26/2011_09_26_drive_0056_sync/image_02/data/0000000132.png",
    "2011_09_26/2011_09_26_drive_0032_sync/image_02/data/0000000207.png",
    "2011_09_26/2011_09_26_drive_0011_sync/image_02/data/0000000157.png",
    "2011_09_26/2011_09_26_drive_0057_sync/image_02/data/0000000319.png",
    "2011_09_26/2011_09_26_drive_0028_sync/image_02/data/0000000303.png",
    "2011_09_26/2011_09_26_drive_0032_sync/image_02/data/0000000350.png",
    "2011_09_26/2011_09_26_drive_0018_sync/image_02/data/0000000096.png",
    "2011_09_26/2011_09_26_drive_0084_sync/image_02/data/0000000238.png",
    "2011_09_26/2011_09_26_drive_0014_sync/image_02/data/0000000213.png",
    "2011_09_26/2011_09_26_drive_0017_sync/image_02/data/0000000010.png",
    "2011_09_26/2011_09_26_drive_0096_sync/image_02/data/0000000381.png",
    "2011_09_26/2011_09_26_drive_0013_sync/image_02/data/0000000040.png",
    "2011_09_26/2011_09_26_drive_0059_sync/image_02/data/0000000137.png",
    "2011_09_26/2011_09_26_drive_0051_sync/image_02/data/0000000312.png",
    "2011_09_26/2011_09_26_drive_0051_sync/image_02/data/0000000379.png",
    "2011_09_26/2011_09_26_drive_0046_sync/image_02/data/0000000062.png",
    "2011_09_26/2011_09_26_drive_0018_sync/image_02/data/0000000046.png",
    "2011_09_26/2011_09_26_drive_0101_sync/image_02/data/0000000457.png",
    "2011_09_26/2011_09_26_drive_0022_sync/image_02/data/0000000654.png",
    "2011_09_29/2011_09_29_drive_0004_sync/image_02/data/0000000094.png",
    "2011_09_26/2011_09_26_drive_0014_sync/image_02/data/0000000030.png",
    "2011_09_26/2011_09_26_drive_0059_sync/image_02/data/0000000290.png",
    "2011_09_26/2011_09_26_drive_0019_sync/image_02/data/0000000087.png",
    "2011_09_28/2011_09_28_drive_0002_sync/image_02/data/0000000343.png",
    "2011_09_26/2011_09_26_drive_0070_sync/image_02/data/0000000069.png",
    "2011_09_26/2011_09_26_drive_0014_sync/image_02/data/0000000141.png",
    "2011_09_26/2011_09_26_drive_0015_sync/image_02/data/0000000209.png",
    "2011_09_29/2011_09_29_drive_0004_sync/image_02/data/0000000285.png",
    "2011_09_26/2011_09_26_drive_0029_sync/image_02/data/0000000123.png",
    "2011_09_26/2011_09_26_drive_0028_sync/image_02/data/0000000118.png",
    "2011_09_26/2011_09_26_drive_0009_sync/image_02/data/0000000414.png",
    "2011_09_26/2011_09_26_drive_0057_sync/image_02/data/0000000125.png",
    "2011_09_26/2011_09_26_drive_0051_sync/image_02/data/0000000230.png",
    "2011_09_26/2011_09_26_drive_0051_sync/image_02/data/0000000292.png",
    "2011_09_26/2011_09_26_drive_0032_sync/image_02/data/0000000360.png",
    "2011_09_26/2011_09_26_drive_0104_sync/image_02/data/0000000015.png",
    "2011_09_26/2011_09_26_drive_0018_sync/image_02/data/0000000076.png",
    "2011_09_29/2011_09_29_drive_0004_sync/image_02/data/0000000036.png",
    "2011_09_26/2011_09_26_drive_0022_sync/image_02/data/0000000634.png",
    "2011_09_26/2011_09_26_drive_0014_sync/image_02/data/0000000060.png",
    "2011_09_26/2011_09_26_drive_0059_sync/image_02/data/0000000310.png",
    "2011_09_26/2011_09_26_drive_0015_sync/image_02/data/0000000229.png",
    "2011_09_26/2011_09_26_drive_0101_sync/image_02/data/0000000175.png",
    "2011_09_26/2011_09_26_drive_0017_sync/image_02/data/0000000040.png",
    "2011_09_26/2011_09_26_drive_0015_sync/image_02/data/0000000187.png",
    "2011_09_26/2011_09_26_drive_0009_sync/image_02/data/0000000384.png",
    "2011_09_26/2011_09_26_drive_0056_sync/image_02/data/0000000201.png",
    "2011_09_26/2011_09_26_drive_0032_sync/image_02/data/0000000114.png",
    "2011_09_26/2011_09_26_drive_0005_sync/image_02/data/0000000010.png",
    "2011_09_29/2011_09_29_drive_0004_sync/image_02/data/0000000162.png",
    "2011_09_26/2011_09_26_drive_0027_sync/image_02/data/0000000103.png",
)

def remove_kitti_files(base_path: str) -> None:
    """
    For each image in KITTI_VAL_RAW, remove any files that match the filename under the corresponding scene folder.
    """
    def delete_matching_files(root: str, target: str) -> None:
        """
        Recursively search for files named target under root and delete them.
        """
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                if fname == target:
                    full_path = os.path.join(dirpath, fname)
                    try:
                        os.remove(full_path)
                        print(f"Removed: {full_path}")
                    except Exception as e:
                        print(f"Error removing {full_path}: {e}")

    for rel_path in KITTI_VAL_RAW:
        parts = rel_path.split('/')
        if len(parts) < 2:
            print(f"Skipping invalid path: {rel_path}")
            continue

        scene = parts[1]
        filename = parts[-1] 
        npy_filename = filename.replace('.png', '.npy') if filename.endswith('.png') else filename

        instances_scene_dir = os.path.join(base_path, "instances_2.5", scene)
        png_scene_dir = os.path.join(base_path, "PNGImage_02", scene)

        if os.path.exists(png_scene_dir):
            delete_matching_files(png_scene_dir, filename)

        if os.path.exists(instances_scene_dir):
            delete_matching_files(instances_scene_dir, filename)
            delete_matching_files(instances_scene_dir, npy_filename)

if __name__ == "__main__":
    base_path="/path/to/your/KITTI"
    remove_kitti_files(base_path)
