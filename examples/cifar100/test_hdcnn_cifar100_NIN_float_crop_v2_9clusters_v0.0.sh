#!/bin/bash
TOOLS=build/tools

# GLOG_logtostderr=1 \
# GLOG_minloglevel=2 \
# $TOOLS/caffe test  \
# 	 --model=models/cifar100_NIN_float_crop_v2/9clusters/9clusters_v0.0/hdcnn_train_test.prototxt \
# 	 --weights=models/cifar100_NIN_float_crop_v2/cifar100_NIN_float_crop_v2_iter_130000.caffemodel,\
# models/cifar100_NIN_float_crop_v2/9clusters/9clusters_v0.0/cluster00_iter_25000.caffemodel,\
# models/cifar100_NIN_float_crop_v2/9clusters/9clusters_v0.0/cluster01_iter_25000.caffemodel,\
# models/cifar100_NIN_float_crop_v2/9clusters/9clusters_v0.0/cluster02_iter_25000.caffemodel,\
# models/cifar100_NIN_float_crop_v2/9clusters/9clusters_v0.0/cluster03_iter_25000.caffemodel,\
# models/cifar100_NIN_float_crop_v2/9clusters/9clusters_v0.0/cluster04_iter_25000.caffemodel,\
# models/cifar100_NIN_float_crop_v2/9clusters/9clusters_v0.0/cluster05_iter_25000.caffemodel,\
# models/cifar100_NIN_float_crop_v2/9clusters/9clusters_v0.0/cluster06_iter_25000.caffemodel,\
# models/cifar100_NIN_float_crop_v2/9clusters/9clusters_v0.0/cluster07_iter_25000.caffemodel,\
# models/cifar100_NIN_float_crop_v2/9clusters/9clusters_v0.0/cluster08_iter_25000.caffemodel\
# 	--iterations=200 --gpu=0


# GLOG_logtostderr=1 \
GLOG_minloglevel=2 \
$TOOLS/caffe test  \
	 --model=models/cifar100_NIN_float_crop_v2/9clusters/9clusters_v0.0/hdcnn_train_test_coarse_consistency_loss.prototxt \
	 --weights=models/cifar100_NIN_float_crop_v2/9clusters/9clusters_v0.0/hdcnn_iter_10000.caffemodel \
	--iterations=200 --gpu=0
