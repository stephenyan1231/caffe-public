#!/bin/bash
TOOLS=build/tools

# GLOG_logtostderr=1 \
GLOG_minloglevel=2 \
$TOOLS/caffe test  \
	 --model=models/cifar100_NIN_float_crop_v2/9clusters/9clusters_v0.0/hdcnn_train_test_coarse_consistency_loss.prototxt \
	 --weights=models/cifar100_NIN_float_crop_v2/9clusters/9clusters_v0.0/hdcnn_iter_10000.caffemodel \
	--iterations=200 --gpu=0
