#!/usr/bin/env sh
# This script converts the cifar data into leveldb format.

BUILD_EXAMPLE=./build/examples/imagenet
DATA=./data/ilsvrc12
TOOLS=./build/tools

TRAIN_DATA_ROOT=/home/zyan3/data/imagenet/ilsvrc12/ILSVRC2012_img_train/
VAL_DATA_ROOT=/home/zyan3/data/imagenet/ilsvrc12/ILSVRC2012_img_val/

echo "Creating database..."

n_clusters=84

fd=imagenet12_VGG_16_layer_${n_clusters}clusters
OUTDIR=/home/zyan3/local/data/imagenet/${fd}
mkdir ${OUTDIR}
OUTDIR=${OUTDIR}/cluster_confusion_mat
INDIR=cluster_confusion_mat_${n_clusters}clusters_VGG_16_layer

#rm -rf ${OUTDIR}
mkdir ${OUTDIR}

imax=`expr ${n_clusters} - 1`
#for i in `seq 0 ${imax}`
# for i in `seq 1 1 40`
for i in `seq 74 1 74`
do
	id=`printf "%02d" $i`
	echo $id

	DB_TR_NM=imagenet_cluster${id}_VGG_16_layer_short_512_train_lmdb	
	rm -rf ${OUTDIR}/${DB_TR_NM} 

	GLOG_logtostderr=1 ${BUILD_EXAMPLE}/convert_imageset_selective_label.bin ${TRAIN_DATA_ROOT} $DATA/train.txt \
	${DATA}/$INDIR/exp_cluster${id}_label_map.txt ${DATA}/$INDIR/train_${id}_compact.txt ${OUTDIR}/${DB_TR_NM} \
	--resize_short_side=512 --shuffle --backend=lmdb

	DB_VAL_NM=imagenet_cluster${id}_VGG_16_layer_short_512_val_lmdb	
	rm -rf ${OUTDIR}/${DB_VAL_NM} 

	GLOG_logtostderr=1 ${BUILD_EXAMPLE}/convert_imageset_selective_label.bin ${VAL_DATA_ROOT} $DATA/val.txt \
	${DATA}/$INDIR/exp_cluster${id}_label_map.txt ${DATA}/$INDIR/val_${id}_compact.txt ${OUTDIR}/${DB_VAL_NM} \
	--resize_short_side=512 --shuffle --backend=lmdb

done

echo "Done."
