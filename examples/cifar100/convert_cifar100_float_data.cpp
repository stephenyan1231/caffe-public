// Zhicheng Yan@eBay
// mostly reuse code from Caffe

#include "boost/scoped_ptr.hpp"
#include <google/protobuf/text_format.h>
#include <glog/logging.h>
//#include <leveldb/db.h>

#include <stdint.h>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <map>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using std::string;
using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

string backend = string("lmdb");
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
		const string& labels_of_interest = "") {
	bool interesting_labels = true;
	bool label_remap = false;

	int n_label = 100, n_label_inte = 0;
  int label_inte,label_newlabel;
  std::vector<bool> label_inte_bit(n_label);
  std::vector<int> label_inte_remap(n_label);

	std::map<int,int> label_2_label;
	if(labels_of_interest == ""){
		interesting_labels = false;
	}
	else{
		std::ifstream labels_file((input_folder + "/" + labels_of_interest).c_str(), std::ios::in);
		string line;
		if(labels_file.is_open()){
			while(getline(labels_file, line)){
				std::stringstream ss(line);
				ss>>label_inte;
				n_label_inte++;
				label_inte_bit[label_inte] = true;
				if(ss>>label_newlabel){
					label_remap = true;
					label_inte_remap[label_inte]=label_newlabel;
				}
			}
			labels_file.close();
		}
		else{
			LOG(ERROR)<<"can not open file "<<input_folder + "/" + labels_of_interest;
		}
	}
	LOG(INFO)<<"interesting_labels:"<<interesting_labels;
  if(label_remap)
	  LOG(INFO)<<"original label is relabeled";

	// Leveldb options
	leveldb::Options options;
	options.create_if_missing = true;
	options.error_if_exists = true;
	// Data buffer
	int fine_label;
	char str_buffer[kCIFARImageNBytes];
	float float_buffer[kCIFARImageNBytes];
	string value;
	caffe::Datum datum;
	datum.set_channels(3);
	datum.set_height(kCIFARSize);
	datum.set_width(kCIFARSize);

	LOG(INFO) << "Writing Training data";


  scoped_ptr<db::DB> train_db(db::GetDB(backend));
  train_db->Open((output_folder + string("/cifar100-train-") + backend).c_str(), db::NEW);
  scoped_ptr<db::Transaction> train_txn(train_db->NewTransaction(false));

//	leveldb::DB* train_db;
//	leveldb::Status status;
//	status = leveldb::DB::Open(options, output_folder + "/cifar100-train-leveldb",
//			&train_db);
//	CHECK(status.ok()) << "Failed to open leveldb.";

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
			if(!interesting_labels || (interesting_labels && label_inte_bit[fine_label])){
				int new_label = fine_label;
				if(interesting_labels && label_remap)
					new_label = label_inte_remap[fine_label];
				datum.set_label(new_label);
				datum.clear_float_data();
				for(int k=0;k<kCIFARImageNBytes;++k)
					datum.add_float_data(float_buffer[k]);
				datum.SerializeToString(&value);
				int length = snprintf(str_buffer, kCIFARImageNBytes, "%d", itemid);
				tr_img_list<<itemid<<" "<<new_label<<std::endl;
				DLOG(INFO)<<"key "<<str_buffer<<" length "<<length;
				train_txn->Put(string(str_buffer, length), value);
//				train_db->Put(leveldb::WriteOptions(), string(str_buffer), value);
			}
		}
		tr_data_file.close();
	}
	train_txn->Commit();
	train_txn.reset(train_db->NewTransaction());

//	tr_data_file.close();
	tr_img_list.close();

	LOG(INFO) << "Writing Testing data";
  scoped_ptr<db::DB> test_db(db::GetDB(backend));
  test_db->Open((output_folder + string("/cifar100-test-") + backend).c_str(), db::NEW);
  scoped_ptr<db::Transaction> test_txn(test_db->NewTransaction(false));
	//	leveldb::DB* test_db;
//	CHECK(
//			leveldb::DB::Open(options, output_folder + "/cifar100-test-leveldb",
//					&test_db).ok()) << "Failed to open leveldb.";
	// Open files
	std::ifstream ts_data_file((input_folder + "/float_test_batch.bin").c_str(), std::ios::in | std::ios::binary);
	std::ofstream ts_img_list((input_folder + "/" + test_list_file).c_str(), std::ios::out);

	CHECK(ts_data_file) << "Unable to open test file.";
	CHECK(ts_img_list) << "Unable to open "<<test_list_file<<" to write.";
	for (itemid = 0; itemid < kCIFARTestSize; ++itemid) {
		read_image(&ts_data_file, &fine_label, float_buffer);
		if(!interesting_labels || (interesting_labels && label_inte_bit[fine_label])){
			int new_label = fine_label;
			if(interesting_labels && label_remap)
				new_label = label_inte_remap[fine_label];
			datum.set_label(new_label);
			datum.clear_float_data();
			for(int k=0;k<kCIFARImageNBytes;++k)
				datum.add_float_data(float_buffer[k]);
			datum.SerializeToString(&value);
			int length = snprintf(str_buffer, kCIFARImageNBytes, "%d", itemid);
			ts_img_list<<itemid<<" "<<new_label<<std::endl;
			test_txn->Put(string(str_buffer, length), value);
//			test_db->Put(leveldb::WriteOptions(), string(str_buffer), value);
		}
	}
	test_txn->Commit();
	test_txn.reset(test_db->NewTransaction());

	ts_data_file.close();
	ts_img_list.close();

//	delete train_db;
//	delete test_db;
}

int main(int argc, char** argv) {
	if (argc != 5 && argc != 6) {
		printf("This script converts the CIFAR 100 dataset to the leveldb format used\n"
				"by caffe to perform classification.\n"
				"Usage:\n"
				"    convert_cifar_float_data input_folder output_folder train_list_file test_list_file labels_of_interest[optional] \n"
				"Where the input folder should contain the binary batch files.\n"
				"The CIFAR dataset could be downloaded at\n"
				"    http://www.cs.toronto.edu/~kriz/cifar.html\n"
				"You should gunzip them after downloading.\n");
	} else {
		google::InitGoogleLogging(argv[0]);
		if(argc == 5)
			convert_dataset(string(argv[1]),string(argv[2]), string(argv[3]), string(argv[4]));
		else
			convert_dataset(string(argv[1]),string(argv[2]), string(argv[3]), string(argv[4]), string(argv[5]));
	}
	return 0;
}
