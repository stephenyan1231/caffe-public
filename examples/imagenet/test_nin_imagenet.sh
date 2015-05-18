#!/usr/bin/env sh

GLOG_logtostderr=1 ./build/tools/caffe test \
--model=models/nin_imagenet/train_val.prototxt \
--weights=models/nin_imagenet/nin_imagenet_train_iter_0.caffemodel \
--iterations=1000 --gpu=0,1

echo "Done."
