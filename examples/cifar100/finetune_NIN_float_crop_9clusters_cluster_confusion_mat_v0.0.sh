#!/usr/bin/env sh

TOOLS=build/tools

NUM_BRANCH=9
VERSION=v0.0

model_dir=models/cifar100_NIN_float_crop_v2
# save_dir=cifar100-float-${NUM_BRANCH}clusters-NIN/cifar100_NIN_float_crop_${NUM_BRANCH}clusters_v1.0
save_dir=${model_dir}/${NUM_BRANCH}clusters/${NUM_BRANCH}clusters_${VERSION}
for i in `seq 0 1 8`
do
	id=`printf "%02d" $i`
	solver=`printf "%s/cluster%02d_solver.prototxt" $save_dir $i`
	solver_lr1=`printf "%s/cluster%02d_solver_lr1.prototxt" $save_dir $i`
	solver_lr2=`printf "%s/cluster%02d_solver_lr2.prototxt" $save_dir $i`
	solverstate1=`printf "%s/cluster%02d_iter_8001.solverstate" $save_dir $i`
	solverstate2=`printf "%s/cluster%02d_iter_15000.solverstate" $save_dir $i`

	echo ${solver}
	echo ${solver_lr1}
	echo ${solver_lr2}
	echo ${solverstate1}
	echo ${solverstate2}

	GLOG_logtostderr=1 $TOOLS/finetune_net_match.bin \
	   --solver=${solver} --match_mode=SUFFIX_MATCH ${model_dir}/cifar100_NIN_float_crop_v2_iter_130000.caffemodel

	 #reduce learning rate by factor of 10
	 GLOG_logtostderr=1 $TOOLS/caffe train \
	   --solver=${solver_lr1} \
	   --snapshot=${solverstate1}

	#reduce learning rate by factor of 10
	 GLOG_logtostderr=1 $TOOLS/caffe train  \
	     --solver=${solver_lr2} \
	     --snapshot=${solverstate2}

done


