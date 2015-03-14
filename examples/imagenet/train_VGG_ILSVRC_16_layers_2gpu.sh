#!/usr/bin/env sh

./build/tools/caffe train --solver=models/VGG_ILSVRC_16_layers/211839e770f7b538e2d8/solver_2gpu.prototxt --weights=models/VGG_ILSVRC_16_layers/211839e770f7b538e2d8/VGG_ILSVRC_16_layers.caffemodel
