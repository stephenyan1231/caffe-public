#!/bin/bash

# GLOG_logtostderr=1 \
GLOG_minloglevel=1 \
./build/tools/caffe train \
--solver=./examples/stanford_background/new_lstm_2d_1_hidden_layer_solver.prototxt \
