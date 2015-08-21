#!/usr/bin/env sh

TOOLS=./build/tools


n_clusters=89
version="v2.0"
model_dir=models/nin_imagenet
save_dir=${model_dir}/${n_clusters}clusters/${n_clusters}clusters_${version}
echo ${save_dir}


imax=`expr ${n_clusters} - 1`
echo ${imax}

for i in `seq 0 ${imax}`
do
	/home/zyan3/proj/caffe_private_hdcnn/
	id=`printf "%02d" $i`
	solver=`printf "%s/cluster%02d_solver.prototxt" $save_dir $i`

	echo ${solver}

	GLOG_logtostderr=1 $TOOLS/finetune_net_match.bin \
	--solver=${solver} --match_mode=SUFFIX_MATCH \
	${model_dir}/nin_imagenet_train_iter_0.caffemodel
done


