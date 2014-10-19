// Copyright 2014 Zhicheng Yan@eBay

#include <stdio.h>  // for snprintf
#include <cuda_runtime.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <google/protobuf/text_format.h>
#include <snappy.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include <string>
#include <vector>
#include <fstream>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

using namespace caffe;
// NOLINT(build/namespaces)

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv);

int main(int argc, char** argv) {
	return feature_extraction_pipeline<float>(argc, argv);
//  return feature_extraction_pipeline<double>(argc, argv);
}

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv) {
	const int num_required_args = 6;
	if (argc < num_required_args) {
		LOG(ERROR)<<
		"This program takes in a trained network and an input data layer, and then"
		" extract features of the input data produced by the net.\n"
		"Usage: extract_features_multilayer_snappy  pretrained_net_param"
		"  feature_extraction_proto_file  extract_feature_blob_name_list"
		"  save_feature_dir_prefix  num_mini_batches  [CPU/GPU]  [DEVICE_ID=0]";
		return 1;
	}
	int arg_pos = num_required_args;

	arg_pos = num_required_args;
	if (argc > arg_pos && strcmp(argv[arg_pos], "GPU") == 0) {
		LOG(ERROR)<< "Using GPU";
		uint device_id = 0;
		if (argc > arg_pos + 1) {
			device_id = atoi(argv[arg_pos + 1]);
			CHECK_GE(device_id, 0);
		}
		LOG(ERROR) << "Using Device_id=" << device_id;
		Caffe::SetDevice(device_id);
		Caffe::set_mode(Caffe::GPU);
	} else {
		LOG(ERROR) << "Using CPU";
		Caffe::set_mode(Caffe::CPU);
	}
	Caffe::set_phase(Caffe::TEST);

	arg_pos = 0;  // the name of the executable
	string pretrained_binary_proto(argv[++arg_pos]);

	// Expected prototxt contains at least one data layer such as
	//  the layer data_layer_name and one feature blob such as the
	//  fc7 top blob to extract features.
	/*
	 layers {
	 name: "data_layer_name"
	 type: DATA
	 data_param {
	 source: "/path/to/your/images/to/extract/feature/images_leveldb"
	 mean_file: "/path/to/your/image_mean.binaryproto"
	 batch_size: 128
	 crop_size: 227
	 mirror: false
	 }
	 top: "data_blob_name"
	 top: "label_blob_name"
	 }
	 layers {
	 name: "drop7"
	 type: DROPOUT
	 dropout_param {
	 dropout_ratio: 0.5
	 }
	 bottom: "fc7"
	 top: "fc7"
	 }
	 */
	string feature_extraction_proto(argv[++arg_pos]);
	shared_ptr<Net<Dtype> > feature_extraction_net(
			new Net<Dtype>(feature_extraction_proto));
	feature_extraction_net->CopyTrainedLayersFrom(pretrained_binary_proto);

	std::ifstream extract_feature_blob_name_list_f(argv[++arg_pos]);
	std::vector<std::string> extract_feature_blob_names;
	string extract_feature_blob_name;
	while (extract_feature_blob_name_list_f >> extract_feature_blob_name) {
		extract_feature_blob_names.push_back(extract_feature_blob_name);
		LOG(INFO)<< "extract blob "<<extract_feature_blob_name<<" feature";
		CHECK(feature_extraction_net->has_blob(extract_feature_blob_name))
																																					<< "Unknown feature blob name "
																																					<< extract_feature_blob_name
																																					<< " in the network "
																																					<< feature_extraction_proto;
	}
	extract_feature_blob_name_list_f.close();
	int num_blobs = extract_feature_blob_names.size();
//
//  string extract_feature_blob_name(argv[++arg_pos]);
//  CHECK(feature_extraction_net->has_blob(extract_feature_blob_name))
//      << "Unknown feature blob name " << extract_feature_blob_name
//      << " in the network " << feature_extraction_proto;

	string save_feature_dir_prefix(argv[++arg_pos]);
	std::vector<string> ftr_dir_names(num_blobs);
	for (int i = 0; i < num_blobs; ++i) {
		string save_feature_dir = save_feature_dir_prefix
				+ string("_") + extract_feature_blob_names[i];
		LOG(INFO)<< "Create folder " << save_feature_dir;
		mkdir(save_feature_dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
		ftr_dir_names[i]=save_feature_dir;
	}

	int num_mini_batches = atoi(argv[++arg_pos]);

	LOG(ERROR)<< "Extacting Features";

	Datum datum;

	const int kMaxKeyStrLength = 100;
	char key_str[kMaxKeyStrLength];
	int num_bytes_of_binary_code = sizeof(Dtype);
	vector<Blob<float>*> input_vec;
	int image_index = 0;
	for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index) {
		LOG(INFO)<<"Batch "<<batch_index;
		feature_extraction_net->Forward(input_vec);
		int image_index_last = image_index;
		const shared_ptr<Blob<Dtype> > label_blob =
		feature_extraction_net->blob_by_name(string("label"));
		Dtype *label_blob_data = label_blob->mutable_cpu_data();

		for (int i = 0; i < num_blobs; ++i) {
			LOG(INFO)<<"[]";
			const shared_ptr<Blob<Dtype> > feature_blob =
			feature_extraction_net->blob_by_name(extract_feature_blob_names[i]);
			int num_features = feature_blob->num();
			int dim_features = feature_blob->count() / num_features;
			Dtype* feature_blob_data;
			image_index = image_index_last;
			for (int n = 0; n < num_features; ++n) {
				if( n%50 == 0) {
					LOG(INFO)<<".";
				}
				datum.set_height(dim_features);
				datum.set_width(1);
				datum.set_channels(1);
				datum.set_label(int(label_blob_data[i]));
				datum.clear_data();
				datum.clear_float_data();
				feature_blob_data = feature_blob->mutable_cpu_data()
				+ feature_blob->offset(n);
				for (int d = 0; d < dim_features; ++d) {
					datum.add_float_data(feature_blob_data[d]);
				}
				string value,compressed_value;
				datum.SerializeToString(&value);
				snappy::Compress(value.data(),value.size(),&compressed_value);
				snprintf(key_str, kMaxKeyStrLength, "%010d", image_index);

				FILE* fp = fopen((ftr_dir_names[i]+string("/")+string(key_str)).c_str(),"wb");
				fwrite(compressed_value.c_str(), 1, compressed_value.size(), fp);
				fclose(fp);

				++image_index;
			}  // for (int n = 0; n < num_features; ++n)
			if (image_index % 1000 == 0) {
				LOG(INFO)<< "<>";
				LOG(ERROR)<< "Extracted "<<extract_feature_blob_names[i] <<" features of " << image_index <<
				" query images.";
			}
		}  // for(int i=0;i<num_blobs;++i){

	}  // for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index)
	// write the last batch
	if (image_index % 1000 != 0) {
		for (int i = 0; i < num_blobs; ++i) {
			LOG(ERROR)<< "Extracted "<< extract_feature_blob_names[i] <<" features of " << image_index <<
			" query images.";
		}
	}

	LOG(ERROR)<< "Successfully extracted the features!";
	return 0;
}
