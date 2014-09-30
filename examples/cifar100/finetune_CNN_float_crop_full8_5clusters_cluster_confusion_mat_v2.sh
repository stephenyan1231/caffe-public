#!/usr/bin/env sh

TOOLS=../../build/tools


for i in `seq 0 4`
do
	id=`printf "%02d" $i`
	solver=`printf "cluster%02d_cifar100_CNN_float_crop_full8_5clusters_v2_solver.prototxt" $i`
	solver_lr1=`printf "cluster%02d_cifar100_CNN_float_crop_full8_5clusters_v2_solver_lr1.prototxt" $i`
	solver_lr2=`printf "cluster%02d_cifar100_CNN_float_crop_full8_5clusters_v2_solver_lr2.prototxt" $i`
	solverstate1=`printf "cluster%02d_cifar100_CNN_float_crop_full8_5clusters_v2_iter_15000.solverstate" $i`
	solverstate2=`printf "cluster%02d_cifar100_CNN_float_crop_full8_5clusters_v2_iter_30000.solverstate" $i`

	echo ${solver}
	echo ${solver_lr1}
	echo ${solver_lr2}
	echo ${solverstate1}
	echo ${solverstate2}

	GLOG_logtostderr=1 $TOOLS/finetune_suffix_match.bin \
	    ${solver} cifar100_float_crop_full8_iter_100000 

	#reduce learning rate by factor of 10
	GLOG_logtostderr=1 $TOOLS/train_net.bin \
	    ${solver_lr1} \
	    ${solverstate1}

	#reduce learning rate by factor of 10
	GLOG_logtostderr=1 $TOOLS/train_net.bin \
	    ${solver_lr2} \
	    ${solverstate2}

done


