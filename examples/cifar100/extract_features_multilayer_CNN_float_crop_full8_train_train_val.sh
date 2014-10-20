#!/bin/bash
TOOLS=../../build/tools
GLOG_logtostderr=1 $TOOLS/extract_features_multilayer_snappy.bin cifar100_CNN_float_crop_full8_train_train_val_iter_100000  cifar100_CNN_float_crop_full8_train_train_val_test.prototxt extract_feature_blob_names.txt decision_CNN/CNN_float_crop_full8/train_val/ 100 GPU 0
