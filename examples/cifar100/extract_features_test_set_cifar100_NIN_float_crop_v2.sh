#!/usr/bin/env sh

GLOG_logtostderr=1 ./build/tools/extract_features \
models/cifar100_NIN_float_crop_v2/cifar100_NIN_float_crop_v2_iter_130000.caffemodel \
models/cifar100_NIN_float_crop_v2/train_test.prototxt \
poolg examples/cifar100/cifar100_NIN_float_crop_v2_test_poolg_leveldb 100 leveldb 1000 GPU 0,1 
