#!/usr/bin/env sh

GLOG_logtostderr=1 ./build/tools/caffe test \
 --model=models/cifar100_NIN_float_crop_v2/train_val/train_test.prototxt \
 --weights=models/cifar100_NIN_float_crop_v2/train_val/cifar100_NIN_float_crop_v2_iter_130000.caffemodel \
 --iterations=100 --gpu=0,1



echo "Done."
