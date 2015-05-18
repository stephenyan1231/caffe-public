#!/usr/bin/env sh
# This scripts downloads the CIFAR10 (binary version) data and unzips it.

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

echo "Downloading..."

for i in `seq 1 5`
do
	file=`printf "https://dl.dropboxusercontent.com/u/44884434/cifar100/float_data_batch_%d.bin" $i`
	wget --no-check-certificate ${file} 
done

wget --no-check-certificate https://dl.dropboxusercontent.com/u/44884434/cifar100/float_test_batch.bin

echo "Done."
