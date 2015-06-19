#!/bin/bash

# GLOG_logtostderr=1 \
GLOG_minloglevel=1 \
./build/tools/caffe train \
--solver=./examples/mnist_semantic_segmentation/lstm_2d_1_hidden_layer_solver.prototxt \
--weights=./examples/mnist_semantic_segmentation/lstm_2d_1_hidden_layer/lstm_2d_1_hidden_layer_iter_1000.caffemodel