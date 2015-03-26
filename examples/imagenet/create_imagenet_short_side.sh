#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs

EXAMPLE=examples/imagenet
DATA=data/ilsvrc12
TOOLS=build/tools

TRAIN_DATA_ROOT=/home/zyan3/data/imagenet/ilsvrc12/ILSVRC2012_img_train/
VAL_DATA_ROOT=/home/zyan3/data/imagenet/ilsvrc12/ILSVRC2012_img_val/

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
  RESIZE_SHORT_SIDE=512
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
  RESIZE_SHORT_SIDE=0
fi

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet training data is stored."
  exit 1
fi

if [ ! -d "$VAL_DATA_ROOT" ]; then
  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
  echo "Set the VAL_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet validation data is stored."
  exit 1
fi

echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --resize_short_side=$RESIZE_SHORT_SIDE \
    --shuffle \
    $TRAIN_DATA_ROOT \
    $DATA/train.txt \
    /home/zyan3/local/data/imagenet/ilsvrc12/ilsvrc12_train_short_512_lmdb

echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --resize_short_side=$RESIZE_SHORT_SIDE \
    $VAL_DATA_ROOT \
    $DATA/val.txt \
    /home/zyan3/local/data/imagenet/ilsvrc12/ilsvrc12_val_short_512_lmdb

echo "Done."
