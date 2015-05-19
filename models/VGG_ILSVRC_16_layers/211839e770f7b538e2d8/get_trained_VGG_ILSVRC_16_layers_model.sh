#!/usr/bin/env sh
# This scripts downloads the CIFAR10 (binary version) data and unzips it.

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

echo "Downloading..."

file=`printf "https://dl.dropboxusercontent.com/u/44884434/2015-HDCNN/caffe_private_hdcnn/models/VGG_ILSVRC_16_layers/211839e770f7b538e2d8/VGG_ILSVRC_16_layers.caffemodel" `
wget --no-check-certificate ${file} 


echo "Done."
