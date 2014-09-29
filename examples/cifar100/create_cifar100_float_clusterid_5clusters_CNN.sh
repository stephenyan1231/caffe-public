#!/usr/bin/env sh
# This script converts the cifar data into leveldb format.

EXAMPLES=../../build/examples/cifar100
DATA=../../data/cifar100
TOOLS=../../build/tools

echo "Creating leveldb..."

leveldb_name=cifar100-float-clusterid-5clusters-CNN-leveldb

rm -rf $leveldb_name
mkdir $leveldb_name

 GLOG_logtostderr=1 $EXAMPLES/convert_cifar100_float_data_clusterid.bin $DATA ./$leveldb_name train_clusterid_5clusters_CNN.txt test_clusterid_5clusters_CNN.txt label_2_clusterid_5clusters_CNN.txt


echo "Done."
