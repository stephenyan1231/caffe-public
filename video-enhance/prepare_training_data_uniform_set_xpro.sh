#!/bin/bash

VIDEO_ENHANCE_BINS=build/video-enhance/tools
OUT_TRAINING_SEGMENT_LDMB=video-enhance/examples/uniform_set_xpro/image-enhance-xpro-train-lmdb

rm -rf $OUT_TRAINING_SEGMENT_LDMB

echo "training."

GLOG_logtostderr=1 $VIDEO_ENHANCE_BINS/train.bin  \
--train_image_list=../dl-image-enhance/data/uniform_set/uniform_set_train_id.txt \
--original_image_ppm_dir=../dl-image-enhance/data/uniform_set/uniform_set_autotone_ppm/ \
--original_image_LAB_lmdb=video-enhance/data/uniform_set/LAB_image_lmdb \
--enhanced_image_LAB_lmdb=video-enhance/data/uniform_set_xpro/LAB_image_lmdb \
--global_feature_lmdb_path=video-enhance/data/uniform_set/global_ftr_lmdb \
--semantic_context_feature_binary_dirs=/home/zyan3/local/proj/caffe_private_hdcnn/examples/uniform_set/VGG_ILSVRC_16_layers_parser_fc7/stitch_fc7_longside_1024/,\
/home/zyan3/local/proj/caffe_private_hdcnn/examples/uniform_set/VGG_ILSVRC_16_layers_parser_fc7/stitch_fc7_longside_2048/ \
--out_training_segment_lmdb=${OUT_TRAINING_SEGMENT_LDMB} 

echo "Done."
