#!/bin/bash
# This script converts the cifar data into leveldb format.

EXAMPLES=../../build/examples/cifar100
DATA=../../data/cifar100
TOOLS=../../build/tools

echo "Creating leveldb..."

rm -rf cifar100-float-train-train-val-leveldb
mkdir cifar100-float-train-train-val-leveldb

GLOG_logtostderr=1 $EXAMPLES/convert_cifar100_float_data_train_train_val.bin $DATA ./cifar100-float-train-train-val-leveldb train_train.txt train_val.txt 40000

echo "Done."
