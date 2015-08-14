#!/bin/bash

CHECKPOINT_DIR=./examples/stanford_background/renet_1_lay

if [ ! -d "$CHECKPOINT_DIR" ]; then
	mkdir $CHECKPOINT_DIR
fi

GLOG_logtostderr=1 \
# GLOG_minloglevel=1 \
./build/tools/caffe train \
--solver=./examples/stanford_background/renet_1_lay_lr-4_solver.prototxt \
--snapshot=./examples/stanford_background/renet_1_lay/renet_1_lay_lr-4_iter_4000.solverstate
