#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs

EXAMPLE=examples/imagenet
DATA=data/ilsvrc12
TOOLS=build/tools

IMAGE_DIR=/home/zyan3/proj/dl-image-enhance/data/uniform_set
IMAGE_DATA_ROOT=${IMAGE_DIR}/uniform_set_autotone_jpg/

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
  RESIZE_SHORT_SIDE=0
  RESIZE_LONG_SIDE=2048
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
  RESIZE_SHORT_SIDE=0
  RESIZE_LONG_SIDE=0
fi

if [ ! -d "$IMAGE_DATA_ROOT" ]; then
  echo "Error: IMAGE_DATA_ROOT is not a path to a directory: $IMAGE_DATA_ROOT"
  echo "Set the IMAGE_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet training data is stored."
  exit 1
fi

echo "Creating lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --resize_short_side=$RESIZE_SHORT_SIDE \
    --resize_long_side=$RESIZE_LONG_SIDE \
    $IMAGE_DATA_ROOT \
    ${IMAGE_DIR}/uniform_set_for_convert_imageset.txt \
    ${IMAGE_DIR}/uniform_set_long_${RESIZE_LONG_SIDE}_lmdb


echo "Done."
