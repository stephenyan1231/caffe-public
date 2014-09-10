#!/usr/bin/env sh

TOOLS=../../build/tools

#GLOG_logtostderr=1 $TOOLS/train_net.bin \
#    cifar100_float_crop_full4_solver.prototxt

GLOG_logtostderr=1 $TOOLS/train_net.bin \
    cifar100_float_crop_full4_solver.prototxt \
    cifar100_float_crop_full4_iter_50000.solverstate


#reduce learning rate by factor of 10
GLOG_logtostderr=1 $TOOLS/train_net.bin \
    cifar100_float_crop_full4_solver_lr1.prototxt \
    cifar100_float_crop_full4_iter_70000.solverstate

#reduce learning rate by factor of 10
GLOG_logtostderr=1 $TOOLS/train_net.bin \
    cifar100_float_crop_full4_solver_lr2.prototxt \
    cifar100_float_crop_full4_iter_85000.solverstate

