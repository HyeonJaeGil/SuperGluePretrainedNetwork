#! /bin/bash

python demo_loop_superglue.py \
--loop_pairs '/media/hj/seagate/datasets/sthereo_training/09/train_sets/positive_pairs.csv' \
--output_dir '/media/hj/seagate/datasets/sthereo_training/09/superglue' \
--resize -1 --superglue 'outdoor' --match_threshold 0.5 --show_keypoints
