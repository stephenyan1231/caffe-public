#!/usr/bin/env sh

TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/train_net.bin \
    cifar100_float_crop_full8/cifar100_float_crop_full8_solver.prototxt

#reduce learning rate by factor of 10
GLOG_logtostderr=1 $TOOLS/train_net.bin \
    cifar100_float_crop_full8/cifar100_float_crop_full8_solver_lr1.prototxt \
    cifar100_float_crop_full8/cifar100_float_crop_full8_iter_70000.solverstate

#reduce learning rate by factor of 10
GLOG_logtostderr=1 $TOOLS/train_net.bin \
    cifar100_float_crop_full8/cifar100_float_crop_full8_solver_lr2.prototxt \
    cifar100_float_crop_full8/cifar100_float_crop_full8_iter_85000.solverstate

