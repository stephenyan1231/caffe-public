#!/usr/bin/env sh

TOOLS=../../build/tools

GLOG_logtostderr=1 ./build/tools/caffe train --solver=models/cifar100_NIN_float_crop_v2/solver.prototxt

GLOG_logtostderr=1 ./build/tools/caffe train --solver=models/cifar100_NIN_float_crop_v2/solver_lr1.prototxt \
 --snapshot=models/cifar100_NIN_float_crop_v2/cifar100_NIN_float_crop_v2_iter_100000.solverstate

GLOG_logtostderr=1 ./build/tools/caffe train --solver=models/cifar100_NIN_float_crop_v2/solver_lr2.prototxt \
 --snapshot=models/cifar100_NIN_float_crop_v2/cifar100_NIN_float_crop_v2_iter_115000.solverstate
