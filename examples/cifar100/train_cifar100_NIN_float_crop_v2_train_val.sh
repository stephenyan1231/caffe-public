#!/usr/bin/env sh

TOOLS=../../build/tools

GLOG_logtostderr=1 ./build/tools/caffe train --solver=models/cifar100_NIN_float_crop_v2/train_val/solver.prototxt

GLOG_logtostderr=1 ./build/tools/caffe train --solver=models/cifar100_NIN_float_crop_v2/train_val/solver_lr1.prototxt \
 --snapshot=models/cifar100_NIN_float_crop_v2/train_val/cifar100_NIN_float_crop_v2_iter_100000.solverstate

GLOG_logtostderr=1 ./build/tools/caffe train --solver=models/cifar100_NIN_float_crop_v2/train_val/solver_lr2.prototxt \
 --snapshot=models/cifar100_NIN_float_crop_v2/train_val/cifar100_NIN_float_crop_v2_iter_115000.solverstate
