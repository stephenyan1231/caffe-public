#!/bin/bash

CHECKPOINT_DIR=./examples/stanford_background/lstm_2d_1_layer

if [ ! -d "$CHECKPOINT_DIR" ]; then
	mkdir $CHECKPOINT_DIR
fi

GLOG_logtostderr=1 \
# GLOG_minloglevel=1 \
./build/tools/caffe train \
--solver=./examples/stanford_background/lstm_2d_1_layer_solver.prototxt \
#--snapshot=./examples/stanford_background/lstm_2d_1_hidden_layer/lstm_2d_1_hidden_layer_iter_500.solverstate \
