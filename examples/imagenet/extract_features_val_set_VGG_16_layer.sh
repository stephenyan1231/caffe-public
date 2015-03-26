#!/usr/bin/env sh

GLOG_logtostderr=1 ./build/tools/extract_features \
models/VGG_ILSVRC_16_layers/211839e770f7b538e2d8/VGG_ILSVRC_16_layers.caffemodel \
models/VGG_ILSVRC_16_layers/211839e770f7b538e2d8/train_val.prototxt \
fc8 examples/imagenet/VGG_16_layer_val_fc8_leveldb 5000 leveldb GPU 0,1

