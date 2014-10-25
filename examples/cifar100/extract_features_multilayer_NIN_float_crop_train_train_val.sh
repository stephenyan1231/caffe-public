#!/bin/bash

cd decision_CNN
mkdir NIN_float_crop
cd NIN_float_crop
mkdir train_val
cd ../..

TOOLS=../../build/tools
GLOG_logtostderr=1 $TOOLS/extract_features_multilayer_snappy.bin cifar100_NIN_float_crop_train_train_val_iter_130000 cifar100_NIN_float_crop_train_train_val_test.prototxt extract_feature_blob_names.txt ./decision_CNN/NIN_float_crop/train_val/ 100 GPU 0
