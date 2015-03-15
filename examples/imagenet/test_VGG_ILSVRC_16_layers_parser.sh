#!/usr/bin/env sh

./build/tools/caffe test --model=models/VGG_ILSVRC_16_layers/211839e770f7b538e2d8/parser_train_val.prototxt --weights=models/VGG_ILSVRC_16_layers/211839e770f7b538e2d8/VGG_ILSVRC_16_layers.caffemodel --gpu=0 --iterations=20
