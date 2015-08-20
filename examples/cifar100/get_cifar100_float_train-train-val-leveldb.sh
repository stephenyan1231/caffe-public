#!/usr/bin/env sh
# This scripts downloads the CIFAR10 (binary version) data and unzips it.

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

echo "Downloading..."


wget --no-check-certificate https://dl.dropboxusercontent.com/u/44884434/2015-hdcnn/caffe_private_hdcnn/examples/cifar100/cifar100-float-train-train-val-leveldb.tar.gz

tar -xzvf cifar100-float-train-train-val-leveldb.tar.gz

echo "Done."
