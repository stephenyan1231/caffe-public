// Zhicheng Yan@eBay
// mostly reuse code from Caffe

#include <google/protobuf/text_format.h>
#include <glog/logging.h>
#include <leveldb/db.h>

#include <stdint.h>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <map>
#include <vector>
#include <algorithm>

#include "caffe/proto/caffe.pb.h"

using std::string;

const int kCIFARSize = 32;
const int kCIFARImageNBytes = 3072;
const int kCIFARTrainSize = 50000;
const int kCIFARTestSize = 10000;
const int kCIFARBatchNum = 5;
const int kCIFARBatchSize = 10000;

void read_image(std::ifstream* file, int* fine_label, float* buffer) {
	char label_char;
	file->read(&label_char, 1);
	*fine_label = label_char;
	file->read((char*) buffer, kCIFARImageNBytes * sizeof(float));
	return;
}

void convert_dataset(const string& input_folder, const string& output_folder,
		const string& train_train_list_file, const string& train_val_list_file,
		int num_train_train) {
	LOG(INFO) << "number of training images from training set:"
			<< num_train_train;
	bool train_train_flag[kCIFARTrainSize];
	memset(train_train_flag, 0, sizeof(bool) * kCIFARTrainSize);
	std::vector<int> rand_nums;
	for (int i = 0; i < kCIFARTrainSize; ++i)
		rand_nums.push_back(i);
	std::random_shuffle(rand_nums.begin(), rand_nums.end());
	for (int i = 0; i < num_train_train; ++i)
		train_train_flag[rand_nums[i]] = true;

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

	LOG(INFO) << "Writing Training and Validation data from Training data";
	leveldb::DB* train_db;
	leveldb::Status status;
	status = leveldb::DB::Open(options,
			output_folder + "/cifar100-train-leveldb", &train_db);
	CHECK(status.ok()) << "Failed to open leveldb.";

	leveldb::DB* val_db;
	CHECK(
			leveldb::DB::Open(options, output_folder + "/cifar100-test-leveldb",
					&val_db).ok()) << "Failed to open leveldb.";

	// Open files
//	std::ifstream tr_data_file((input_folder + "/train.bin").c_str(),
//			std::ios::in | std::ios::binary);
	std::ofstream tr_tr_img_list(
			(input_folder + "/" + train_train_list_file).c_str(),
			std::ios::out);
	std::ofstream tr_val_img_list(
			(input_folder + "/" + train_val_list_file).c_str(), std::ios::out);


	CHECK(tr_tr_img_list) << "Unable to open " << train_train_list_file
			<< " to write";
	CHECK(tr_val_img_list) << "Unable to open " << train_val_list_file
			<< " to write.";
	int itemid = 0;
	for (int i = 0; i < kCIFARBatchNum; ++i) {
		snprintf(str_buffer, kCIFARImageNBytes, "/float_data_batch_%d.bin",
				i + 1);
		std::ifstream tr_data_file((input_folder + string(str_buffer)).c_str(),
				std::ios::in | std::ios::binary);
		CHECK(tr_data_file) << "Unable to open train set";
		for (int j = 0; j < kCIFARBatchSize; ++j, ++itemid) {
			read_image(&tr_data_file, &fine_label, float_buffer);
			datum.set_label(fine_label);
			datum.clear_float_data();
			for (int k = 0; k < kCIFARImageNBytes; ++k)
				datum.add_float_data(float_buffer[k]);
			datum.SerializeToString(&value);
			snprintf(str_buffer, kCIFARImageNBytes, "%05d", itemid);
			if (train_train_flag[itemid]) {
				tr_tr_img_list << itemid << " " << fine_label << std::endl;
				train_db->Put(leveldb::WriteOptions(), string(str_buffer),
						value);
			} else {
				tr_val_img_list << itemid << " " << fine_label << std::endl;
				val_db->Put(leveldb::WriteOptions(), string(str_buffer), value);
			}
		}
		tr_data_file.close();
	}
//	tr_data_file.close();
	tr_tr_img_list.close();
	tr_val_img_list.close();

	delete train_db;
	delete val_db;
}

int main(int argc, char** argv) {
	if (argc != 6) {
		printf(
				"This script converts the CIFAR 100 dataset to the leveldb format used\n"
						"by caffe to perform classification.\n"
						"Usage:\n"
						"    convert_cifar100_float_data_train_train_val input_folder output_folder train_train_list_file train_val_list_file train_train_img_num\n"
						"Where the input folder should contain the binary batch files.\n"
						"The CIFAR dataset could be downloaded at\n"
						"    http://www.cs.toronto.edu/~kriz/cifar.html\n"
						"You should gunzip them after downloading.\n");
	} else {
		google::InitGoogleLogging(argv[0]);
		convert_dataset(string(argv[1]), string(argv[2]), string(argv[3]),
				string(argv[4]), atoi(argv[5]));
	}
	return 0;
}
