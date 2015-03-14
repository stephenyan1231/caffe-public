#!/usr/bin/env sh

TOOLS=./build/tools

GLOG_logtostderr=1 $TOOLS/extract_features examples/cifar10/cifar10_quick_2gpu_iter_5000.caffemodel \
examples/cifar10/cifar10_quick_train_test.prototxt conv2 examples/cifar10/quick_2gpu_conv2_leveldb 50 \
leveldb GPU 0,1
