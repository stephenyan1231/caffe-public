#!/usr/bin/env sh
# This scripts downloads the CIFAR10 (binary version) data and unzips it.

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

echo "Downloading..."


wget --no-check-certificate https://dl.dropboxusercontent.com/u/44884434/2015-hdcnn/caffe_private_hdcnn/examples/imagenet/hdcnn/NIN/train_val_pred_prob.npy

echo "Done."
