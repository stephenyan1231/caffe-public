#!/usr/bin/env sh

GLOG_logtostderr=1 ./build/tools/extract_features models/VGG_ILSVRC_16_layers/211839e770f7b538e2d8/VGG_ILSVRC_16_layers.caffemodel models/VGG_ILSVRC_16_layers/211839e770f7b538e2d8/parser_fc7_train_val.prototxt stitch_fc7 examples/imagenet/VGG_ILSVRC_16_layers_parser_fc7/stitch_fc7_leveldb 10 leveldb GPU 0
