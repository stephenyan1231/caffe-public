// Copyright 2014 Zhicheng Yan.
//
// This script converts the CIFAR float-typed dataset with global constrast normalization and ZCA whitening to the leveldb format used
// by caffe to perform classification.
// Usage:
//    convert_cifar_float_data input_folder output_db_file

#include <google/protobuf/text_format.h>
#include <glog/logging.h>
#include <leveldb/db.h>

#include <stdint.h>
#include <fstream>  // NOLINT(readability/streams)
#include <string>

#include "caffe/proto/caffe.pb.h"

using std::string;

const int kCIFARSize = 32;
const int kCIFARImageNBytes = 3072;
const int kCIFARBatchSize = 1400000/5;
const int kCIFARTestSize = 10000;
const int kCIFARTrainBatches = 5;

void read_image(std::ifstream* file, int* label, float* buffer) {
  char label_char;
  file->read(&label_char, 1);
  *label = label_char;
  file->read((char*)buffer, kCIFARImageNBytes * sizeof(float));
  return;
}

void convert_dataset(const string& input_folder, const string& output_folder) {
  // Leveldb options
  leveldb::Options options;
  options.create_if_missing = true;
  options.error_if_exists = true;
  // Data buffer
  int label;
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
  status = leveldb::DB::Open(options, output_folder + "/cifar-train-leveldb",
      &train_db);
  CHECK(status.ok()) << "Failed to open leveldb.";
  for (int fileid = 0; fileid < kCIFARTrainBatches; ++fileid) {
    // Open files
    LOG(INFO) << "Training Batch " << fileid + 1;
    snprintf(str_buffer, kCIFARImageNBytes, "/float_data_batch_%d.bin", fileid + 1);
    std::ifstream data_file((input_folder + str_buffer).c_str(),
        std::ios::in | std::ios::binary);
    CHECK(data_file) << "Unable to open train file #" << fileid + 1;
    for (int itemid = 0; itemid < kCIFARBatchSize; ++itemid) {
      read_image(&data_file, &label, float_buffer);
      datum.set_label(label);
      datum.clear_float_data();
      for(int i=0;i<kCIFARImageNBytes;++i)
    	  datum.add_float_data(float_buffer[i]);
      datum.SerializeToString(&value);
      snprintf(str_buffer, kCIFARImageNBytes, "%010d",
          fileid * kCIFARBatchSize + itemid);
      train_db->Put(leveldb::WriteOptions(), string(str_buffer), value);
    }
  }

  LOG(INFO) << "Writing Testing data";
  leveldb::DB* test_db;
  CHECK(leveldb::DB::Open(options, output_folder + "/cifar-test-leveldb",
      &test_db).ok()) << "Failed to open leveldb.";
  // Open files
  std::ifstream data_file((input_folder + "/float_test_batch.bin").c_str(),
      std::ios::in | std::ios::binary);
  CHECK(data_file) << "Unable to open test file.";
  for (int itemid = 0; itemid < kCIFARTestSize; ++itemid) {
    read_image(&data_file, &label, float_buffer);
    datum.set_label(label);
    datum.clear_float_data();
    for(int i=0;i<kCIFARImageNBytes;++i)
    	datum.add_float_data(float_buffer[i]);
    datum.SerializeToString(&value);
    snprintf(str_buffer, kCIFARImageNBytes, "%010d", itemid);
    test_db->Put(leveldb::WriteOptions(), string(str_buffer), value);
  }

  delete train_db;
  delete test_db;
}

int main(int argc, char** argv) {
  if (argc != 3) {
    printf("This script converts the CIFAR dataset to the leveldb format used\n"
           "by caffe to perform classification.\n"
           "Usage:\n"
           "    convert_cifar_float_data input_folder output_folder\n"
           "Where the input folder should contain the binary batch files.\n"
           "The CIFAR dataset could be downloaded at\n"
           "    http://www.cs.toronto.edu/~kriz/cifar.html\n"
           "You should gunzip them after downloading.\n");
  } else {
    google::InitGoogleLogging(argv[0]);
    convert_dataset(string(argv[1]), string(argv[2]));
  }
  return 0;
}
