#!/usr/bin/env sh

TOOLS=./build/tools


n_clusters=84
version="v0.0"
model_dir=models/VGG_ILSVRC_16_layers/211839e770f7b538e2d8
save_dir=${model_dir}/${n_clusters}clusters/${n_clusters}clusters_${version}
echo ${save_dir}

imax=`expr ${n_clusters} - 1`
echo ${imax}

# for i in `seq 0 ${imax}`
for i in `seq 32 1 32`
do
	/home/zyan3/proj/caffe_private_hdcnn/
	id=`printf "%02d" $i`
	solver=`printf "%s/cluster%02d_solver_multiscale_train_2gpu.prototxt" $save_dir $i`
	snapshot=`printf "%s/cluster%02d_iter_2000.solverstate" $save_dir $i`


	echo ${solver}

	GLOG_logtostderr=1 $TOOLS/finetune_net_match.bin \
	    --solver=${solver} --match_mode=SUFFIX_MATCH \
	    ${model_dir}/VGG_ILSVRC_16_layers.caffemodel

#	GLOG_logtostderr=1 $TOOLS/caffe train \
#	    --solver=${solver} --snapshot=${snapshot}

done


