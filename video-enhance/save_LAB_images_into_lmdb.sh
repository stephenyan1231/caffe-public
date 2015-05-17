#!/bin/bash

# original images
SAVE_LAB_IMAGE_LMDB=/home/zyan3/proj/caffe_private_video_enhance/video-enhance/data/uniform_set/LAB_image_lmdb
rm -rf ${SAVE_LAB_IMAGE_LMDB}

python video-enhance/python/export_LAB_image_into_binary_file.py \
--image_list_file=/home/zyan3/proj/dl-image-enhance/data/uniform_set/uniform_set.txt \
--tiff_image_dir=/home/zyan3/proj/dl-image-enhance/data/uniform_set/uniform_set_autotone_tif/ \
--save_LAB_image_lmdb=${SAVE_LAB_IMAGE_LMDB}


# xpro images
SAVE_LAB_IMAGE_LMDB=/home/zyan3/proj/caffe_private_video_enhance/video-enhance/data/uniform_set_xpro/LAB_image_lmdb
rm -rf ${SAVE_LAB_IMAGE_LMDB}

python video-enhance/python/export_LAB_image_into_binary_file.py \
--image_list_file=/home/zyan3/proj/dl-image-enhance/data/uniform_set/uniform_set.txt \
--tiff_image_dir=/home/zyan3/proj/dl-image-enhance/data/uniform_set_xpro/uniform_set_xpro_tif/ \
--save_LAB_image_lmdb=${SAVE_LAB_IMAGE_LMDB}


echo "Done."
