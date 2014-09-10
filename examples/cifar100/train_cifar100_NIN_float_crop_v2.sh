#!/usr/bin/env sh

TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/train_net.bin \
    cifar100_NIN_float_crop_v2_solver.prototxt

#reduce learning rate by factor of 10
GLOG_logtostderr=1 $TOOLS/train_net.bin \
    cifar100_NIN_float_crop_v2_solver_lr1.prototxt \
    cifar100_NIN_float_crop_v2_iter_100000.solverstate

#reduce learning rate by factor of 10
GLOG_logtostderr=1 $TOOLS/train_net.bin \
    cifar100_NIN_float_crop_v2_solver_lr2.prototxt \
    cifar100_NIN_float_crop_v2_iter_115000.solverstate
