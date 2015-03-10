#!/usr/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe train \
  --solver=examples/cifar10/cifar10_quick_2gpu_solver.prototxt

$TOOLS/caffe train \
  --solver=examples/cifar10/cifar10_quick_2gpu_solver_lr1.prototxt \
  --snapshot=examples/cifar10/cifar10_quick_2gpu_iter_4000.solverstate
