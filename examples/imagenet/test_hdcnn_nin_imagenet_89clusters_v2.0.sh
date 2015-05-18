#!/bin/bash
TOOLS=build/tools

# GLOG_logtostderr=1 \
GLOG_minloglevel=2 \
$TOOLS/caffe test  \
	 --model=models/nin_imagenet/89clusters/89clusters_v2.0/hdcnn_train_val.prototxt \
	 --weights=models/nin_imagenet/89clusters/89clusters_v2.0/hdcnn_iter_0.caffemodel \
	 	--iterations=1000 --gpu=0



# # GLOG_logtostderr=1 \
# GLOG_minloglevel=2 \
# $TOOLS/caffe test  \
# 	 --model=/home/zyan3/proj/caffe_private_hdcnn/models/nin_imagenet/89clusters/89clusters_v2.0/hdcnn_train_val.prototxt \
# 	 --weights=/home/zyan3/proj/caffe_private_hdcnn/models/nin_imagenet/nin_imagenet_train_iter_0.caffemodel,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster00_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster01_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster02_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster03_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster04_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster05_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster06_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster07_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster08_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster09_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster10_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster11_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster12_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster13_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster14_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster15_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster16_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster17_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster18_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster19_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster20_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster21_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster22_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster23_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster24_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster25_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster26_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster27_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster28_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster29_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster30_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster31_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster32_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster33_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster34_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster35_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster36_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster37_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster38_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster39_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster40_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster41_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster42_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster43_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster44_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster45_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster46_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster47_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster48_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster49_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster50_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster51_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster52_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster53_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster54_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster55_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster56_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster57_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster58_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster59_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster60_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster61_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster62_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster63_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster64_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster65_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster66_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster67_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster68_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster69_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster70_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster71_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster72_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster73_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster74_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster75_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster76_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster77_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster78_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster79_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster80_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster81_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster82_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster83_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster84_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster85_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster86_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster87_imagenet12_NIN_89clusters_v2.0_iter_40000,\
# /home/zyan3/proj/caffe_private_decision-cnn/examples/imagenet/imagenet12_NIN_89clusters/v2.0/cluster88_imagenet12_NIN_89clusters_v2.0_iter_40000 \
# 	--iterations=500 --gpu=0,1
