#!/usr/bin/env sh

TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/train_net.bin \
    cifar100_CNN_float_crop_full8_clusterid_5clusters_solver.prototxt

#reduce learning rate by factor of 10
GLOG_logtostderr=1 $TOOLS/train_net.bin \
     cifar100_CNN_float_crop_full8_clusterid_5clusters_solver_lr1.prototxt\
    cifar100_CNN_float_crop_full8_clusterid_5clusters_iter_70000.solverstate

#reduce learning rate by factor of 10
GLOG_logtostderr=1 $TOOLS/train_net.bin \
     cifar100_CNN_float_crop_full8_clusterid_5clusters_solver_lr2.prototxt\
    cifar100_CNN_float_crop_full8_clusterid_5clusters_iter_90000.solverstate
