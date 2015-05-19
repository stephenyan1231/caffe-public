#!/usr/bin/env sh
# This scripts downloads the CIFAR10 (binary version) data and unzips it.

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

echo "Downloading..."

file=`printf "https://dl.dropboxusercontent.com/u/44884434/2015-HDCNN/caffe_private_hdcnn/models/VGG_ILSVRC_16_layers/211839e770f7b538e2d8/84clusters/84clusters_v2.0/iter_12000.tar.gz"`

wget --no-check-certificate ${file}
tar -xzvf iter_12000.tar.gz



for i in `seq 0 83`
do
	file=`printf "https://dl.dropboxusercontent.com/u/44884434/2015-HDCNN/caffe_private_hdcnn/models/VGG_ILSVRC_16_layers/211839e770f7b538e2d8/84clusters/84clusters_v2.0/cluster%02d_iter_12000.caffemodel" $i`
	wget --no-check-certificate ${file} 
done

echo "Done."


