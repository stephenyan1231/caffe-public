#include <stdio.h>  // for snprintf
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
using boost::shared_ptr;
using std::string;
namespace db = caffe::db;

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv);

int main(int argc, char** argv) {
  return feature_extraction_pipeline<float>(argc, argv);
//  return feature_extraction_pipeline<double>(argc, argv);
}

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  const int num_required_args = 7;
  if (argc < num_required_args) {
    LOG(ERROR)<<
    "This program takes in a trained network and an input data layer, and then"
    " extract features of the input data produced by the net.\n"
    "Usage: extract_features  pretrained_net_param"
    "  feature_extraction_proto_file  extract_feature_blob_name1[,name2,...]"
    "  save_feature_dataset_name1[,name2,...]  num_mini_batches  db_type "
    "  [commit_frequency] [CPU/GPU] [DEVICE_ID=0]\n"
    "Note: you can extract multiple features in one pass by specifying"
    " multiple feature blob names and dataset names seperated by ','."
    " The names cannot contain white space characters and the number of blobs"
    " and datasets must be equal.";
    return 1;
  }
  int arg_pos = num_required_args;
  int commit_frequency = 10;

  arg_pos = num_required_args;
  if (argc > arg_pos) {
    commit_frequency = atoi(argv[arg_pos]);
    if (argc > arg_pos + 1 && strcmp(argv[arg_pos + 1], "GPU") == 0) {
      LOG(ERROR)<< "Using GPU";
      std::vector<int> device_id;
      if (argc > arg_pos + 2) {
        device_id = caffe::parse_int_list(std::string(argv[arg_pos + 2]));
        for(int i=0;i<device_id.size();++i) {
          CHECK_GE(device_id[i], 0);
          LOG(ERROR) << "Using Device " << device_id[i];
        }
      }
      //    LOG(ERROR) << "Using Device_id=" << device_id;
      Caffe::set_mode(Caffe::GPU);
      Caffe::InitDevices(device_id);
    } else {
      LOG(ERROR) << "Using CPU";
      Caffe::set_mode(Caffe::CPU);
    }
  }
  LOG(INFO)<<"database commit frequency "<<commit_frequency;

  Caffe::set_phase(Caffe::TEST);

  arg_pos = 0;  // the name of the executable
  std::string pretrained_binary_proto(argv[++arg_pos]);
  std::vector<std::string> all_pretrained_binary_proto;
  boost::split(all_pretrained_binary_proto, pretrained_binary_proto,
      boost::is_any_of(","));

  std::string feature_extraction_proto(argv[++arg_pos]);
  shared_ptr<Net<Dtype> > feature_extraction_net(
      new Net<Dtype>(feature_extraction_proto));
  feature_extraction_net->PostInit();
  feature_extraction_net->CopyTrainedLayersFrom(all_pretrained_binary_proto);
  for (int i = 0; i < all_pretrained_binary_proto.size(); ++i) {
    LOG(ERROR)<<"Use weights from model "<<all_pretrained_binary_proto[i];
  }

  std::string extract_feature_blob_names(argv[++arg_pos]);
  std::vector<std::string> blob_names;
  boost::split(blob_names, extract_feature_blob_names, boost::is_any_of(","));

  std::string save_feature_dataset_names(argv[++arg_pos]);
  std::vector<std::string> dataset_names;
  boost::split(dataset_names, save_feature_dataset_names,
      boost::is_any_of(","));
  CHECK_EQ(blob_names.size(), dataset_names.size())<<
  " the number of blob names and dataset names must be equal";
  size_t num_features = blob_names.size();

  for (size_t i = 0; i < num_features; i++) {
    CHECK(feature_extraction_net->has_blob(blob_names[i]))
                                                              << "Unknown feature blob name "
                                                              << blob_names[i]
                                                              << " in the network "
                                                              << feature_extraction_proto;
  }

  int num_mini_batches = atoi(argv[++arg_pos]);

  std::vector<shared_ptr<db::DB> > feature_dbs;
  std::vector<shared_ptr<db::Transaction> > txns;
  char *db_type = argv[++arg_pos];
  for (size_t i = 0; i < num_features; ++i) {
    LOG(INFO)<< "Opening dataset " << dataset_names[i];
    shared_ptr<db::DB> db(db::GetDB(std::string(db_type)));
    db->Open(dataset_names.at(i), db::NEW);
    feature_dbs.push_back(db);
    shared_ptr<db::Transaction> txn(db->NewTransaction(false));
    txns.push_back(txn);
  }

  LOG(ERROR)<< "Extracting Features";

  int replica_num = Caffe::GetReplicasNum();
  Datum datum;
  const int kMaxKeyStrLength = 1024;
  char key_str[kMaxKeyStrLength];
  std::vector<Blob<float>*> input_vec;
  std::vector<int> image_indices(num_features, 0);
  for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index) {
    LOG(INFO)<<"batch index "<<batch_index;
    feature_extraction_net->Forward(input_vec);
    for (int i = 0; i < num_features; ++i) {
      for(int r = 0; r < replica_num; ++r) {
        const shared_ptr<Blob<Dtype> > feature_blob = feature_extraction_net
        ->blob_by_name(blob_names[i], r);
        LOG(INFO)<<"feature_blob shape "<<feature_blob->num()<<" "
        <<feature_blob->channels()<<" "
        <<feature_blob->height()<<" "
        <<feature_blob->width()<<" "
        <<feature_blob->count() * sizeof(Dtype);
        int batch_size = feature_blob->num();
        int dim_features = feature_blob->count() / batch_size;
        int channels = feature_blob->channels();
        int height = feature_blob->height();
        int width = feature_blob->width();
        const Dtype* feature_blob_data;
        for (int n = 0; n < batch_size; ++n) {
          datum.set_height(height);
          datum.set_width(width);
          datum.set_channels(channels);
          datum.clear_data();
          datum.clear_float_data();
          feature_blob_data = feature_blob->cpu_data() +
          feature_blob->offset(n);
          for (int d = 0; d < dim_features; ++d) {
            if(d == 0 && n == (batch_size / 2)) {
              LOG(INFO)<<"feature_blob_data[0]:"<<feature_blob_data[0];
            }
            datum.add_float_data(feature_blob_data[d]);
          }
          int length = snprintf(key_str, kMaxKeyStrLength, "%d",
              image_indices[i]);
          string out;
          CHECK(datum.SerializeToString(&out));
          txns.at(i)->Put(std::string(key_str, length), out);

          ++image_indices[i];
          if (image_indices[i] % commit_frequency == 0) {
            txns.at(i)->Commit();
            txns.at(i).reset(feature_dbs.at(i)->NewTransaction());
            LOG(ERROR)<< "Extracted features of " << image_indices[i] <<
            " query images for feature blob " << blob_names[i];
          }
        }  // for (int n = 0; n < batch_size; ++n)
      } // for(int r = 0; r < replica_num; ++r) {
    }  // for (int i = 0; i < num_features; ++i)
    feature_extraction_net->clear_blobs_gpu();
  }  // for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index)
  // write the last batch
  for (int i = 0; i < num_features; ++i) {
    if (image_indices[i] % commit_frequency != 0) {
      txns.at(i)->Commit();
    }
    LOG(ERROR)<< "Extracted features of " << image_indices[i] <<
    " query images for feature blob " << blob_names[i];
    feature_dbs.at(i)->Close();
  }

  LOG(ERROR)<< "Successfully extracted the features!";
  return 0;
}

