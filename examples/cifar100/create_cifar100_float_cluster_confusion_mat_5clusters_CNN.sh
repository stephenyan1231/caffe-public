#!/usr/bin/env sh
# This script converts the cifar data into leveldb format.

EXAMPLES=../../build/examples/cifar100
DATA=../../data/cifar100
TOOLS=../../build/tools

echo "Creating leveldb..."

OUTDIR=cluster_confusion_mat_fulllabel_5clusters_CNN
INDIR=cluster_confusion_mat_5clusters_CNN

rm -rf ${OUTDIR}
mkdir ${OUTDIR}

for i in `seq 0 4`
do
	id=`printf "%02d" $i`
	echo $id
	rm -rf ${OUTDIR}/cifar100_float_cluster${id}_leveldb 
	mkdir ${OUTDIR}/cifar100_float_cluster${id}_leveldb

	GLOG_logtostderr=1 $EXAMPLES/convert_cifar100_float_data.bin $DATA ${OUTDIR}/cifar100_float_cluster${id}_leveldb $INDIR/train_${id}.txt $INDIR/test_${id}.txt $INDIR/cluster${id}_labels.txt 0


done

echo "Done."
