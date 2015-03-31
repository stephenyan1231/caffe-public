#!/usr/bin/env sh


layer_name=stitch_fc7
working_dir=examples/imagenet/VGG_ILSVRC_16_layers_parser_fc7
mkdir ${working_dir}
rm -rf ${working_dir}/${layer_name}_leveldb

GLOG_logtostderr=1 ./build/tools/extract_features models/VGG_ILSVRC_16_layers/211839e770f7b538e2d8/VGG_ILSVRC_16_layers.caffemodel \
 models/VGG_ILSVRC_16_layers/211839e770f7b538e2d8/parser_fc7_train_val_downsample.prototxt  \
 stitch_fc7 examples/imagenet/VGG_ILSVRC_16_layers_parser_fc7/stitch_fc7_leveldb 100 leveldb GPU 0
