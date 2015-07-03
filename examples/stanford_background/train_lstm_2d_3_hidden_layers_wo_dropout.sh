#!/bin/bash

CHECKPOINT_DIR=./examples/stanford_background/lstm_2d_3_hidden_layers

if [ ! -d "$CHECKPOINT_DIR" ]; then
	mkdir $CHECKPOINT_DIR
fi

GLOG_logtostderr=1 \
# GLOG_minloglevel=1 \
./build/tools/caffe train \
--solver=./examples/stanford_background/lstm_2d_3_hidden_layers_wo_dropout_solver.prototxt \
#--snapshot=./examples/stanford_background/lstm_2d_3_hidden_layers/lstm_2d_3_hidden_layers_wo_dropout_iter_500.solverstate \
