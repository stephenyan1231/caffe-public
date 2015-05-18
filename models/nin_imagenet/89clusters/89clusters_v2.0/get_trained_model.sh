#!/usr/bin/env sh
# This scripts downloads the CIFAR10 (binary version) data and unzips it.

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

echo "Downloading..."

file=`printf "https://dl.dropboxusercontent.com/u/44884434/2015-HDCNN/caffe_private_hdcnn/models/nin_imagenet/89clusters/89clusters_v2.0/hdcnn_iter_0.caffemodel" `
wget --no-check-certificate ${file} 

file=`printf "https://dl.dropboxusercontent.com/u/44884434/2015-HDCNN/caffe_private_hdcnn/models/nin_imagenet/89clusters/89clusters_v2.0/iter_40000.tar.gz" `
wget --no-check-certificate ${file} 

tar -xzvf iter_40000.tar.gz

echo "Done."
