#!/usr/bin/env sh

TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/test_net.bin cifar100_NIN_float_crop_v2_test.prototxt cifar100_NIN_float_crop_v2_iter_130000 100 GPU 0

echo "Done."
