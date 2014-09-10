#!/usr/bin/env sh
# This script converts the cifar data into leveldb format.

EXAMPLES=../../build/examples/cifar10
DATA=../../data/cifar100
TOOLS=../../build/tools

echo "Creating leveldb..."

rm -rf cifar100-float-leveldb
mkdir cifar100-float-leveldb

GLOG_logtostderr=1 $EXAMPLES/convert_cifar_float_data.bin $DATA ./cifar100-float-leveldb

echo "Computing image mean..."

GLOG_logtostderr=1 $TOOLS/compute_image_mean.bin ./cifar100-float-leveldb/cifar-train-leveldb float_mean.binaryproto

echo "Done."
