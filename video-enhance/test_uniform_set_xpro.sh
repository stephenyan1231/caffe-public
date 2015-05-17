#!/bin/bash

VIDEO_ENHANCE_BINS=build/video-enhance/tools

OUTPUT_IMAGE_DIR=video-enhance/examples/uniform_set_xpro/enhanced_images/
mkdir ${OUTPUT_IMAGE_DIR}

echo "testing."

GLOG_logtostderr=1 $VIDEO_ENHANCE_BINS/test.bin  \
--test_image_list=../dl-image-enhance/data/uniform_set/uniform_set_test_id.txt \
--original_image_ppm_dir=../dl-image-enhance/data/uniform_set/uniform_set_autotone_ppm/ \
--original_image_LAB_lmdb=video-enhance/data/uniform_set/LAB_image_lmdb \
--enhanced_image_LAB_lmdb=video-enhance/data/uniform_set_xpro/LAB_image_lmdb \
--global_feature_lmdb_path=video-enhance/data/uniform_set/global_ftr_lmdb \
--semantic_context_feature_model=models/VGG_ILSVRC_16_layers/211839e770f7b538e2d8/parser_fc7_train_val_downsample_deploy.prototxt  \
--semantic_context_feature_weights=models/VGG_ILSVRC_16_layers/211839e770f7b538e2d8/VGG_ILSVRC_16_layers.caffemodel \
--video_enhance_model=models/dl-image-enhance-2-layer-192/deploy.prototxt \
--video_enhance_weights=models/dl-image-enhance-2-layer-192/dl-image-enhance-2-layer-192-train_iter_1600000.caffemodel \
--gpu=0 \
--global_feature_mean_file=video-enhance/data/uniform_set/global_feature_mean.binaryproto \
--semantic_context_feature_mean_file=video-enhance/data/uniform_set/semantic_context_feature_mean.binaryproto \
--pixel_feature_mean_file=video-enhance/data/uniform_set/pixel_feature_mean.binaryproto \
--output_image_dir=${OUTPUT_IMAGE_DIR} \


echo "Done."
