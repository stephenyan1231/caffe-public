#!/usr/bin/env sh

TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/train_net.bin \
   cifar100_CNN_float_crop_full8_train_train_val_solver.prototxt 

#reduce learning rate by factor of 10
GLOG_logtostderr=1 $TOOLS/train_net.bin \
    cifar100_CNN_float_crop_full8_train_train_val_solver_lr1.prototxt\
    cifar100_CNN_float_crop_full8_train_train_val_iter_70000.solverstate

#reduce learning rate by factor of 10
GLOG_logtostderr=1 $TOOLS/train_net.bin \
    cifar100_CNN_float_crop_full8_train_train_val_solver_lr2.prototxt \
   cifar100_CNN_float_crop_full8_train_train_val_iter_85000.solverstate 

