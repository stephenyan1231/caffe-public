#!/usr/bin/env sh
# This script converts the cifar data into leveldb format.

EXAMPLES=build/examples/cifar100
DATA=data/cifar100
TOOLS=build/tools

echo "Creating leveldb..."

NUM_BRANCH=9


OUTDIR=${CAFFE_LOCAL_PROJ_DIR}examples/cifar100/cifar100-float-${NUM_BRANCH}clusters-NIN
echo ${OUTDIR}
mkdir ${OUTDIR}
OUTDIR=${OUTDIR}/cluster_confusion_mat_${NUM_BRANCH}clusters_NIN

INDIR=cluster_confusion_mat_${NUM_BRANCH}clusters_NIN

rm -rf ${OUTDIR}
mkdir ${OUTDIR}

for i in `seq 0 8`
do
	id=`printf "%02d" $i`
	echo $id
	DB_NM=cifar100_float_cluster${id}_NIN_leveldb
	rm -rf ${OUTDIR}/${DB_NM} 
	mkdir ${OUTDIR}/${DB_NM}

	GLOG_logtostderr=1 $EXAMPLES/convert_cifar100_float_data.bin $DATA ${OUTDIR}/${DB_NM} $INDIR/train_${id}.txt $INDIR/test_${id}.txt $INDIR/exp_cluster${id}_label_map.txt 


done

echo "Done."
