#!/bin/bash

IMAGE_DIR=/usr/local/google/home/zyan/data/stanford_background/images/
LABEL_DIR=/usr/local/google/home/zyan/data/stanford_background/labels/
TRAIN_FILE_LIST=./examples/stanford_background/train_0.txt
TEST_FILE_LIST=./examples/stanford_background/test_0.txt
TRAIN_DB_NAME=./examples/stanford_background/train_0_lmdb
TEST_DB_NAME=./examples/stanford_background/test_0_lmdb

rm -r $TRAIN_DB_NAME
GLOG_logtostderr=1 ./build/examples/stanford_background/create_stanford_background_data.bin \
--min_height=240 --min_width=321 \
$IMAGE_DIR $LABEL_DIR $TRAIN_FILE_LIST $TRAIN_DB_NAME

rm -r $TEST_DB_NAME
GLOG_logtostderr=1 ./build/examples/stanford_background/create_stanford_background_data.bin \
--min_height=240 --min_width=321 \
$IMAGE_DIR $LABEL_DIR $TEST_FILE_LIST $TEST_DB_NAME