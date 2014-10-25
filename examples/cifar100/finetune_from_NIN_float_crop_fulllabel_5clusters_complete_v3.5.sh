#!/bin/bash

TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/finetune_net_prefix_match.bin \
    cifar100_NIN_float_crop_gating_5clusters_v3.5_solver.prototxt \
cifar100_NIN_float_crop_clusterid_5clusters_iter_200000 \
cifar100_NIN_float_crop_v2_iter_130000

GLOG_logtostderr=1 $TOOLS/train_net.bin \
   cifar100_NIN_float_crop_gating_5clusters_v3.5_solver_lr1.prototxt cifar100_NIN_float_crop_gating_5clusters_v3.5_finetune_from_NIN_float_crop_fulllabel_complete_iter_5000.solverstate


GLOG_logtostderr=1 $TOOLS/train_net.bin \
   cifar100_NIN_float_crop_gating_5clusters_v3.5_solver_lr2.prototxt cifar100_NIN_float_crop_gating_5clusters_v3.5_finetune_from_NIN_float_crop_fulllabel_complete_iter_15000.solverstate




echo "Done."
