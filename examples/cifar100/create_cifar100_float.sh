#!/usr/bin/env sh
# This script converts the cifar data into leveldb format.

EXAMPLES=./build/examples/cifar100
DATA=./data/cifar100
TOOLS=./build/tools

echo "Creating database..."

rm -rf examples/cifar100/cifar100-float-lmdb
mkdir examples/cifar100/cifar100-float-lmdb

GLOG_logtostderr=1 $EXAMPLES/convert_cifar100_float_data.bin $DATA ./examples/cifar100/cifar100-float-lmdb train.txt test.txt

echo "Computing image mean..."

# GLOG_logtostderr=1 $TOOLS/compute_image_mean.bin ./examples/cifar100/cifar100-float-leveldb/cifar-train-leveldb float_mean.binaryproto

echo "Done."
