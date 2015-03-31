#!/usr/bin/env sh
example_dir=examples/imagenet
dn_name=VGG_16_layer_val_fc8_lmdb
rm -rf ${example_dir}/${dn_name}

GLOG_logtostderr=1 ./build/tools/extract_features \
models/VGG_ILSVRC_16_layers/211839e770f7b538e2d8/VGG_ILSVRC_16_layers.caffemodel \
models/VGG_ILSVRC_16_layers/211839e770f7b538e2d8/train_val.prototxt \
fc8 ${example_dir}/${dn_name} 50 lmdb 100 GPU 0,1

