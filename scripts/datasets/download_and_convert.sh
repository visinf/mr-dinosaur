#!/bin/bash

case $1 in

  movi_e)
    echo "Creating movi_e webdataset in outputs/movi_e"
    SEED=23894734
    # training set
    mkdir -p outputs/movi_e
    mkdir -p outputs/movi_e/train
    python conversion_scripts/convert_tfds.py movi_e/128x128:1.0.0 train outputs/movi_e/train --dataset_path gs://kubric-public/tfds
    # eval set
    mkdir -p outputs/movi_e/val
    python conversion_scripts/convert_tfds.py movi_e/128x128:1.0.0 validation outputs/movi_e/val --dataset_path gs://kubric-public/tfds
    # test set
    mkdir -p outputs/movi_e/test
    python conversion_scripts/convert_tfds.py movi_e/128x128:1.0.0 test outputs/movi_e/test --dataset_path gs://kubric-public/tfds
    # pseudo set
    mkdir -p outputs/movi_e/train_with_label
    python conversion_scripts/convert_pseudo.py --train_image_dir data/movi_e/train_pseudo_smurf/PNGImages --train_label_dir data/movi_e/train_pseudo_smurf/instances_2.5 outputs/movi_e/train_with_label --seed $SEED
    ;;

  KITTI_pre)
    echo "Downloading and reorganizing KITTI in data/KITTI"
    SEED=23894734
    # Download KITTI dataset
    save_path="data/KITTI/KITTI_train"
    bash ./download_scripts/download_kitti.sh "$save_path"
    # Reorganize KITTI dataset
    python ./download_scripts/kitti_reorganize.py --source_dir $save_path
    ;;

  KITTI)
    echo "Converting KITTI to webdataset in outputs/KITTI"
    SEED=23894734
    # training set
    mkdir -p outputs/kitti/train
    python conversion_scripts/convert_kitti.py --train_image_dir data/KITTI/KITTI_train outputs/kitti/train --seed $SEED
    # eval set
    mkdir -p outputs/kitti/val
    python conversion_scripts/convert_kitti.py --val_image_dir data/KITTI/KITTI_test/image_2 --val_label_dir data/KITTI/KITTI_test/instance outputs/kitti/val --seed $SEED
    # pseudo set
    mkdir -p outputs/kitti/train_with_label_01
    python conversion_scripts/convert_kitti.py --train_image_dir data/KITTI/KITTI_smurf_1.7_PNGImages_02/PNGImages_02 --train_label_dir data/KITTI/KITTI_smurf_1.7_PNGImages_02/instances_2.5 outputs/kitti/train_with_label_01 --seed $SEED
    mkdir -p outputs/kitti/train_with_label_02
    python conversion_scripts/convert_kitti.py --train_image_dir data/KITTI/KITTI_smurf_1.7_PNGImages_03/PNGImages_03 --train_label_dir data/KITTI/KITTI_smurf_1.7_PNGImages_03/instances_2.5 outputs/kitti/train_with_label_02 --seed $SEED
    mkdir -p outputs/kitti/train_with_label
    python conversion_scripts/combine_several_folders.py outputs/kitti/train_with_label_01 outputs/kitti/train_with_label_02 outputs/kitti/train_with_label
    ;;
    
  TRI-PD_pre)
    echo "Reorganizing TRI-PD in data/TRI_PD"
    SEED=23894734
    train_path="data/TRI_PD/PD_simplified"
    val_path="data/TRI_PD/pd_test_video"
    # reorganize training set of TRI-PD 
    python ./download_scripts/pd_reorganize.py --mode train --base_path $train_path
    # reorganize validation set of TRI-PD 
    python ./download_scripts/pd_reorganize.py --mode val --base_path $val_path
    ;;

  TRI-PD)
    echo "Converting TRI-PD to webdataset in data/TRI_PD"
    SEED=23894734
    # training set
    mkdir -p outputs/pd/train
    python conversion_scripts/convert_pd.py --train_image_dir data/TRI_PD/PD_simplified outputs/pd/train --seed $SEED
    # eval set
    mkdir -p outputs/pd/val
    python conversion_scripts/convert_pd.py --val_image_dir data/TRI_PD/pd_test_video/rgb --val_label_dir data/TRI_PD/pd_test_video/ari_masks outputs/pd/val --seed $SEED
    # pseudo set
    mkdir -p outputs/pd/train_with_label_01
    python conversion_scripts/convert_pd.py --train_image_dir data/TRI_PD/PD_smurf_0.5_PNGImages_01/PNGImages_01 --train_label_dir data/TRI_PD/PD_smurf_0.5_PNGImages_01/instances_2.5 outputs/pd/train_with_label_01 --seed $SEED
    mkdir -p outputs/pd/train_with_label_05
    python conversion_scripts/convert_pd.py --train_image_dir data/TRI_PD/PD_smurf_0.5_PNGImages_05/PNGImages_05 --train_label_dir data/TRI_PD/PD_smurf_0.5_PNGImages_05/instances_2.5 outputs/pd/train_with_label_05 --seed $SEED
    mkdir -p outputs/pd/train_with_label_06
    python conversion_scripts/convert_pd.py --train_image_dir data/TRI_PD/PD_smurf_0.5_PNGImages_06/PNGImages_06 --train_label_dir data/TRI_PD/PD_smurf_0.5_PNGImages_06/instances_2.5 outputs/pd/train_with_label_06 --seed $SEED
    mkdir -p outputs/pd/train_with_label
    python conversion_scripts/combine_several_folders.py outputs/pd/train_with_label_01 outputs/pd/train_with_label_05 outputs/pd/train_with_label_06 outputs/pd/train_with_label
    ;;

  Custom)
    SEED=23894734
    IMG_DIR="${2:-data/custom/custom_test}"
    echo "Converting Custom dataset (test only) from '$IMG_DIR' to webdataset in outputs/custom"
    mkdir -p outputs/custom/test
    python conversion_scripts/convert_custom.py --test_image_dir "$IMG_DIR" outputs/custom/test --seed "$SEED"
    ;;

  *)
    echo "Unknown dataset $1"
    echo "Only KITTI_pre, KITTI, TRI-PD_pre, TRI-PD, movi_e and Custom are supported."
    ;;
esac
