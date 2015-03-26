#!/usr/bin/env sh

GLOG_logtostderr=1 ./build/tools/extract_features \
models/cifar100_NIN_float_crop_v2/train_val/cifar100_NIN_float_crop_v2_iter_130000.caffemodel \
models/cifar100_NIN_float_crop_v2/train_val/train_test.prototxt \
poolg examples/cifar100/train_val/cifar100_NIN_float_crop_v2_val_poolg_leveldb 100 leveldb GPU 0,1 
