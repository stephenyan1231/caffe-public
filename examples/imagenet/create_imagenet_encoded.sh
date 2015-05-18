#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs

EXAMPLE=examples/imagenet
DATA=data/ilsvrc12
TOOLS=build/tools

TRAIN_DATA_ROOT=/home/zyan3/local/data/imagenet/ilsvrc12/ILSVRC2012_img_train/
VAL_DATA_ROOT=/home/zyan3/local/data/imagenet/ilsvrc12/ILSVRC2012_img_val/


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

echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --encoded \
    $VAL_DATA_ROOT \
    $DATA/val.txt \
    /home/zyan3/local/data/imagenet/ilsvrc12/ilsvrc12_val_encoded_lmdb

 echo "Creating train lmdb..."

 GLOG_logtostderr=1 $TOOLS/convert_imageset \
      --shuffle --encoded \
      $TRAIN_DATA_ROOT \
      $DATA/train.txt \
      /home/zyan3/local/data/imagenet/ilsvrc12/ilsvrc12_train_encoded_lmdb



echo "Done."
