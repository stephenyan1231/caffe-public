// Zhicheng Yan@eBay
// mostly reuse code from Caffe

#include <google/protobuf/text_format.h>
#include <glog/logging.h>
#include <leveldb/db.h>

#include <stdint.h>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <map>

#include "caffe/proto/caffe.pb.h"

using std::string;

const int kCIFARSize = 32;
const int kCIFARImageNBytes = 3072;
const int kCIFARTrainSize = 50000;
const int kCIFARTestSize = 10000;
const int kCIFARBatchNum = 5;
const int kCIFARBatchSize = 10000;

void read_image(std::ifstream* file, int* fine_label,
		float* buffer) {
	char label_char;
	file->read(&label_char, 1);
	*fine_label = label_char;
	file->read((char*)buffer, kCIFARImageNBytes*sizeof(float));
	return;
}

void convert_dataset(const string& input_folder, const string& output_folder,
		const string& train_list_file, const string& test_list_file,
		const string& label_2_clusterid_file) {

	std::map<int, int> label_2_clusterid;
	std::ifstream label_2_clusterid_fp((input_folder + "/" + label_2_clusterid_file).c_str(), std::ios::in);
	string line;
	int label_c = 0;
	if(label_2_clusterid_fp.is_open()){
		while(getline(label_2_clusterid_fp, line)){
			label_2_clusterid[label_c++] = atoi(line.c_str());
		}
		label_2_clusterid_fp.close();
	}
	else{
		LOG(ERROR)<<"can not open file "<<label_2_clusterid_file;
	}
	LOG(INFO)<<"read "<<label_2_clusterid.size()<<" label_to_clusterid mapping";

	// Leveldb options
	leveldb::Options options;
	options.create_if_missing = true;
	options.error_if_exists = true;
	// Data buffer
	int coarse_label, fine_label;
	char str_buffer[kCIFARImageNBytes];
	float float_buffer[kCIFARImageNBytes];
	string value;
	caffe::Datum datum;
	datum.set_channels(3);
	datum.set_height(kCIFARSize);
	datum.set_width(kCIFARSize);

	LOG(INFO) << "Writing Training data";
	leveldb::DB* train_db;
	leveldb::Status status;
	status = leveldb::DB::Open(options, output_folder + "/cifar100-train-leveldb",
			&train_db);
	CHECK(status.ok()) << "Failed to open leveldb.";

	// Open files
//	std::ifstream tr_data_file((input_folder + "/train.bin").c_str(), std::ios::in | std::ios::binary);
	std::ofstream tr_img_list((input_folder + "/" + train_list_file).c_str(), std::ios::out);
//	CHECK(tr_data_file) << "Unable to open train set";
	CHECK(tr_img_list) << "Unable to open "<< train_list_file <<" to write";

	int itemid=0;
	for(int i=0;i<kCIFARBatchNum;++i){
		snprintf(str_buffer, kCIFARImageNBytes, "/float_data_batch_%d.bin",
				i + 1);
		std::ifstream tr_data_file((input_folder + string(str_buffer)).c_str(),
				std::ios::in | std::ios::binary);
		CHECK(tr_data_file) << "Unable to open train set";
		for(int j=0;j<kCIFARBatchSize;++j,++itemid){
			read_image(&tr_data_file, &fine_label, float_buffer);
			datum.clear_float_data();
			for(int k=0;k<kCIFARImageNBytes;++k)
				datum.add_float_data(float_buffer[k]);
			datum.set_label(label_2_clusterid[fine_label]);
			datum.SerializeToString(&value);
			snprintf(str_buffer, kCIFARImageNBytes, "%05d", itemid);
			tr_img_list<<itemid<<" "<<label_2_clusterid[fine_label]<<std::endl;
			train_db->Put(leveldb::WriteOptions(), string(str_buffer), value);
		}
		tr_data_file.close();
	}
//	tr_data_file.close();
	tr_img_list.close();

	LOG(INFO) << "Writing Testing data";
	leveldb::DB* test_db;
	CHECK(
			leveldb::DB::Open(options, output_folder + "/cifar100-test-leveldb",
					&test_db).ok()) << "Failed to open leveldb.";
	// Open files
	std::ifstream ts_data_file((input_folder + "/float_test_batch.bin").c_str(), std::ios::in | std::ios::binary);
	std::ofstream ts_img_list((input_folder + "/" + test_list_file).c_str(), std::ios::out);

	CHECK(ts_data_file) << "Unable to open test file.";
	CHECK(ts_img_list) << "Unable to open "<<test_list_file<<" to write.";
	for (int itemid = 0; itemid < kCIFARTestSize; ++itemid) {
		read_image(&ts_data_file, &fine_label, float_buffer);
		datum.set_label(label_2_clusterid[fine_label]);
		datum.clear_float_data();
		for(int k=0;k<kCIFARImageNBytes;++k)
			datum.add_float_data(float_buffer[k]);
		datum.SerializeToString(&value);
		snprintf(str_buffer, kCIFARImageNBytes, "%05d", itemid);
		ts_img_list<<itemid<<" "<<label_2_clusterid[fine_label]<<std::endl;
		test_db->Put(leveldb::WriteOptions(), string(str_buffer), value);
	}
	ts_data_file.close();
	ts_img_list.close();

	delete train_db;
	delete test_db;
}

int main(int argc, char** argv) {
	if (argc != 6) {
		printf("This script converts the CIFAR 100 dataset to the leveldb format used\n"
				"by caffe to perform classification.\n"
				"Usage:\n"
				"    convert_cifar_data input_folder output_folder train_list_file test_list_file label_2_clusterid_file\n"
				"Where the input folder should contain the binary batch files.\n"
				"The CIFAR dataset could be downloaded at\n"
				"    http://www.cs.toronto.edu/~kriz/cifar.html\n"
				"You should gunzip them after downloading.\n");
	} else {
		google::InitGoogleLogging(argv[0]);
		convert_dataset(string(argv[1]),string(argv[2]), string(argv[3]), string(argv[4]), string(argv[5]));
	}
	return 0;
}
