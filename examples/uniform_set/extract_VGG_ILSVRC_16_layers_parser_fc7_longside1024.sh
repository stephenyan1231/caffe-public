#!/usr/bin/env sh


layer_name=stitch_fc7
config=longside_1024
save_dir=/home/zyan3/local/proj/caffe_private_hdcnn/examples/uniform_set/VGG_ILSVRC_16_layers_parser_fc7
mkdir ${save_dir}
# db_name=${layer_name}_${config}_lmdb
# rm -rf ${save_dir}/${db_name}

output_dir=${layer_name}_${config}
rm -rf ${save_dir}/${output_dir}
mkdir ${save_dir}/${output_dir}

# GLOG_logtostderr=1 ./build/tools/extract_features_binary_output models/VGG_ILSVRC_16_layers/211839e770f7b538e2d8/VGG_ILSVRC_16_layers.caffemodel \
#  models/VGG_ILSVRC_16_layers/211839e770f7b538e2d8/parser_fc7_train_val_downsample.prototxt  \
#  stitch_fc7 ${save_dir}/${db_name} 10  lmdb 1 GPU 0,1

# GLOG_minloglevel=2
GLOG_logtostderr=1 ./build/tools/extract_features_binary_output models/VGG_ILSVRC_16_layers/211839e770f7b538e2d8/VGG_ILSVRC_16_layers.caffemodel \
 models/VGG_ILSVRC_16_layers/211839e770f7b538e2d8/parser_fc7_train_val_downsample_longside1024.prototxt  \
 stitch_fc7 ${save_dir}/${output_dir}/ 115  lmdb 2 GPU 0
