#!/usr/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe test \
  --model=examples/cifar10/cifar10_quick_train_test.prototxt \
  --weights=examples/cifar10/cifar10_quick_2gpu_iter_5000.caffemodel \
  --gpu=0,1


