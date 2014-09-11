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
		const string& labels_of_interest = "", bool label_2_label_flag = true) {
	LOG(INFO)<<"label_2_label_flag:"<<label_2_label_flag;
	bool interesting_labels = true;
	std::map<int,int> label_2_label;
	if(labels_of_interest == ""){
		interesting_labels = false;
	}
	else{
		std::ifstream labels_file((input_folder + "/" + labels_of_interest).c_str(), std::ios::in);
		string line;
		if(labels_file.is_open()){
			int new_label = 0;
			while(getline(labels_file, line)){
				label_2_label[atoi(line.c_str())] = new_label++;
			}
			labels_file.close();
		}
		else{
			LOG(ERROR)<<"can not open file "<<input_folder + "/" + labels_of_interest;
		}
	}
	LOG(INFO)<<"interesting_labels:"<<interesting_labels;

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
			if(!interesting_labels || (interesting_labels && label_2_label.find(fine_label) != label_2_label.end())){
				int new_label = fine_label;
				if(interesting_labels && label_2_label_flag)
					new_label = label_2_label[fine_label];
				datum.set_label(new_label);
				datum.clear_float_data();
				for(int k=0;k<kCIFARImageNBytes;++k)
					datum.add_float_data(float_buffer[k]);
				datum.SerializeToString(&value);
				snprintf(str_buffer, kCIFARImageNBytes, "%05d", itemid);
				tr_img_list<<itemid<<" "<<new_label<<std::endl;
				train_db->Put(leveldb::WriteOptions(), string(str_buffer), value);
			}
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
		if(!interesting_labels || (interesting_labels && label_2_label.find(fine_label) != label_2_label.end())){
			int new_label = fine_label;
			if(interesting_labels && label_2_label_flag)
				new_label = label_2_label[fine_label];
			datum.set_label(new_label);
			datum.clear_float_data();
			for(int k=0;k<kCIFARImageNBytes;++k)
				datum.add_float_data(float_buffer[k]);
			datum.SerializeToString(&value);
			snprintf(str_buffer, kCIFARImageNBytes, "%05d", itemid);
			ts_img_list<<itemid<<" "<<new_label<<std::endl;
			test_db->Put(leveldb::WriteOptions(), string(str_buffer), value);
		}
	}
	ts_data_file.close();
	ts_img_list.close();

	delete train_db;
	delete test_db;
}

int main(int argc, char** argv) {
	if (argc != 5 && argc != 6 && argc != 7) {
		printf("This script converts the CIFAR 100 dataset to the leveldb format used\n"
				"by caffe to perform classification.\n"
				"Usage:\n"
				"    convert_cifar_float_data input_folder output_folder train_list_file test_list_file labels_of_interest[optional] label_2_label_flag[optional]\n"
				"Where the input folder should contain the binary batch files.\n"
				"The CIFAR dataset could be downloaded at\n"
				"    http://www.cs.toronto.edu/~kriz/cifar.html\n"
				"You should gunzip them after downloading.\n");
	} else {
		google::InitGoogleLogging(argv[0]);
		if(argc == 5)
			convert_dataset(string(argv[1]),string(argv[2]), string(argv[3]), string(argv[4]));
		else if(argc == 6)
			convert_dataset(string(argv[1]),string(argv[2]), string(argv[3]), string(argv[4]), string(argv[5]));
		else
			convert_dataset(string(argv[1]),string(argv[2]), string(argv[3]), string(argv[4]), string(argv[5]), atoi(argv[6]));
	}
	return 0;
}
