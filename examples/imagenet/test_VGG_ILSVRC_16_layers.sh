#!/usr/bin/env sh

# GLOG_logtostderr=1 \
GLOG_minloglevel=2 \
./build/tools/caffe test --model=models/VGG_ILSVRC_16_layers/211839e770f7b538e2d8/train_val_multiscale_train.prototxt \
--weights=models/VGG_ILSVRC_16_layers/211839e770f7b538e2d8/VGG_ILSVRC_16_layers.caffemodel \
--iterations=20 --gpu=1

echo "Done."
