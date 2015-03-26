#!/bin/bash
TOOLS=build/tools
GLOG_logtostderr=1 $TOOLS/finetune_net_match \
	--solver=/home/zyan3/proj/caffe_private_hdcnn/models/cifar100_NIN_float_crop_v2/9clusters/9clusters_v0.0/hdcnn_solver.prototxt \
	/home/zyan3/proj/caffe_private_hdcnn/models/cifar100_NIN_float_crop_v2/cifar100_NIN_float_crop_v2_iter_130000.caffemodel \
	/home/zyan3/proj/caffe_private_hdcnn/models/cifar100_NIN_float_crop_v2/9clusters/9clusters_v0.0/cluster00_iter_25000.caffemodel \
	/home/zyan3/proj/caffe_private_hdcnn/models/cifar100_NIN_float_crop_v2/9clusters/9clusters_v0.0/cluster01_iter_25000.caffemodel \
	/home/zyan3/proj/caffe_private_hdcnn/models/cifar100_NIN_float_crop_v2/9clusters/9clusters_v0.0/cluster02_iter_25000.caffemodel \
	/home/zyan3/proj/caffe_private_hdcnn/models/cifar100_NIN_float_crop_v2/9clusters/9clusters_v0.0/cluster03_iter_25000.caffemodel \
	/home/zyan3/proj/caffe_private_hdcnn/models/cifar100_NIN_float_crop_v2/9clusters/9clusters_v0.0/cluster04_iter_25000.caffemodel \
	/home/zyan3/proj/caffe_private_hdcnn/models/cifar100_NIN_float_crop_v2/9clusters/9clusters_v0.0/cluster05_iter_25000.caffemodel \
	/home/zyan3/proj/caffe_private_hdcnn/models/cifar100_NIN_float_crop_v2/9clusters/9clusters_v0.0/cluster06_iter_25000.caffemodel \
	/home/zyan3/proj/caffe_private_hdcnn/models/cifar100_NIN_float_crop_v2/9clusters/9clusters_v0.0/cluster07_iter_25000.caffemodel \
	/home/zyan3/proj/caffe_private_hdcnn/models/cifar100_NIN_float_crop_v2/9clusters/9clusters_v0.0/cluster08_iter_25000.caffemodel \

# GLOG_logtostderr=1 $TOOLS/caffe train \
# 	--solver=/home/zyan3/proj/caffe_private_hdcnn/models/cifar100_NIN_float_crop_v2/9clusters/9clusters_v0.0/hdcnn_solver_lr1.prototxt \
# 	--weights=/home/zyan3/proj/caffe_private_hdcnn/models/cifar100_NIN_float_crop_v2/9clusters/9clusters_v0.0/hdcnn_iter_10000.caffemodel \

# GLOG_logtostderr=1 $TOOLS/caffe train \
# 	--solver=/home/zyan3/proj/caffe_private_hdcnn/models/cifar100_NIN_float_crop_v2/9clusters/9clusters_v0.0/hdcnn_solver_lr2.prototxt \
# 	--weights=/home/zyan3/proj/caffe_private_hdcnn/models/cifar100_NIN_float_crop_v2/9clusters/9clusters_v0.0/hdcnn_iter_20000.caffemodel \
