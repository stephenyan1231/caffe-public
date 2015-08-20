// Copyright 2014 BVLC and contributors.
// This program converts a set of images to a leveldb by storing them as Datum
// proto buffers.
// Usage:
//   convert_imageset [-g] ROOTFOLDER/ LISTFILE DB_NAME RANDOM_SHUFFLE[0 or 1]
//                     [resize_height] [resize_width]
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.JPEG 7
//   ....
// if RANDOM_SHUFFLE is 1, a random shuffle will be carried out before we
// process the file lines.
// Optional flag -g indicates the images should be read as
// single-channel grayscale. If omitted, grayscale images will be
// converted to color.
#include "boost/scoped_ptr.hpp"
#include <glog/logging.h>
#include <gflags/gflags.h>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

//#include <leveldb/db.h>
//#include <leveldb/write_batch.h>
//#include <lmdb.h>
#include <sys/stat.h>

#include <algorithm>
#include <iostream>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>


using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using std::string;
using boost::scoped_ptr;

DEFINE_bool(gray, false,
    "When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false,
    "Randomly shuffle the order of images and their labels");
DEFINE_string(backend, "lmdb",
        "The backend {lmdb, leveldb} for storing the result");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");
DEFINE_int32(resize_short_side, 0, "Size of short side images are resized to");
DEFINE_int32(resize_long_side, 0, "Size of long side images are resized to");
DEFINE_bool(check_size, false,
    "When this option is on, check that all the datum have the same size");
DEFINE_bool(encoded, false,
    "When this option is on, the encoded image will be save in datum");

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc != 6) {
    printf("Convert a set of images to the leveldb format used\n"
        "as input for Caffe.\n"
        "Usage:\n"
        "    convert_imageset_selective_label ROOTFOLDER/ LISTFILE IN_LABELFILE OUT_LISTFILE DB_NAME"
        " \n"
        "The ImageNet dataset for the training demo is at\n"
        "    http://www.image-net.org/download-images\n");
    return 1;
  }

  const bool is_color = !FLAGS_gray;
  const bool check_size = FLAGS_check_size;

  std::ifstream infile(argv[2]);
  std::vector<std::pair<string, int> > lines;
  string filename;
  int label;
  while (infile >> filename >> label) {
    lines.push_back(std::make_pair(filename, label));
  }
  if (FLAGS_shuffle) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    std::random_shuffle(lines.begin(), lines.end());
  }
  LOG(INFO) << "A total of " << lines.size() << " images.";


  int n_label = 1000, n_label_inte = 0;
  std::ifstream inlabelfile(argv[3]);
  int label_inte,label_newlabel;
  std::vector<bool> label_inte_bit(n_label);
  std::vector<int> label_inte_remap(n_label);
  bool label_remap = false;
  std::string line;
  while(std::getline(inlabelfile,line)){
	  std::stringstream ss(line);
	  ss>>label_inte;
	  n_label_inte++;
	  label_inte_bit[label_inte]=true;
	  if(ss>>label_newlabel){
		  label_remap = true;
		  label_inte_remap[label_inte] = label_newlabel;
	  }
  }
  if(label_remap)
	  LOG(INFO)<<"original label is relabeled";

//  while(inlabelfile>>label_inte){
//	  n_label_inte++;
//	  label_inte_bit[label_inte]=true;
//  }
  inlabelfile.close();
  LOG(INFO)<<n_label_inte<<" labels in the cluster";

//  string db_backend = "leveldb";
//  if (argc >= (8)) {
//    db_backend = string(argv[7]);
//    if (!(db_backend == "leveldb") && !(db_backend == "lmdb")) {
//      LOG(FATAL) << "Unknown db backend " << db_backend;
//    }
//  }

//  int resize_height = 0;
//  int resize_width = 0;
//  if (argc >= (arg_offset+9)) {
//    resize_height = atoi(argv[arg_offset+8]);
//    LOG(INFO)<<"resize height "<<resize_height;
//  }
//  if (argc >= (arg_offset+10)) {
//    resize_width = atoi(argv[arg_offset+9]);
//    LOG(INFO)<<"resize width "<<resize_width;
//  }

  int resize_height = std::max<int>(0, FLAGS_resize_height);
  int resize_width = std::max<int>(0, FLAGS_resize_width);
  int resize_short_side = std::max<int>(0, FLAGS_resize_short_side);
  int resize_long_side = std::max<int>(0, FLAGS_resize_long_side);
  CHECK(!(resize_short_side > 0 && resize_long_side))
  <<"at most one of resize_short_side and resize_long_side can be non-zero";
  if(resize_height > 0 && resize_width > 0){
  	LOG(INFO)<<"resize_height "<<resize_height<<" resize_width "<<resize_width;
  }
  if(resize_short_side > 0){
  	LOG(INFO)<<"resize_short_side "<<resize_short_side;
  }

  // Create new DB
  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
  db->Open(argv[5], db::NEW);
  scoped_ptr<db::Transaction> txn(db->NewTransaction());

//  // Open new db
//  // lmdb
//  MDB_env *mdb_env;
//  MDB_dbi mdb_dbi;
//  MDB_val mdb_key, mdb_data;
//  MDB_txn *mdb_txn;
//  // leveldb
//  leveldb::DB* db;
//  leveldb::Options options;
//  options.error_if_exists = true;
//  options.create_if_missing = true;
//  options.write_buffer_size = 268435456;
//  leveldb::WriteBatch* batch;

//  // Open db
//  if (db_backend == "leveldb") {  // leveldb
//    LOG(INFO) << "Opening leveldb " << argv[5];
//    leveldb::Status status = leveldb::DB::Open(
//        options, argv[5], &db);
//    CHECK(status.ok()) << "Failed to open leveldb " << argv[5];
//    batch = new leveldb::WriteBatch();
//  } else if (db_backend == "lmdb") {  // lmdb
//    LOG(INFO) << "Opening lmdb " << argv[5];
//    CHECK_EQ(mkdir(argv[5], 0744), 0)
//        << "mkdir " << argv[5] << "failed";
//    CHECK_EQ(mdb_env_create(&mdb_env), MDB_SUCCESS) << "mdb_env_create failed";
//    CHECK_EQ(mdb_env_set_mapsize(mdb_env, 1099511627776), MDB_SUCCESS)  // 1TB
//        << "mdb_env_set_mapsize failed";
//    CHECK_EQ(mdb_env_open(mdb_env, argv[5], 0, 0664), MDB_SUCCESS)
//        << "mdb_env_open failed";
//    CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)
//        << "mdb_txn_begin failed";
//    CHECK_EQ(mdb_open(mdb_txn, NULL, 0, &mdb_dbi), MDB_SUCCESS)
//        << "mdb_open failed";
//  } else {
//    LOG(FATAL) << "Unknown db backend " << db_backend;
//  }

  std::ofstream out_listfile(argv[4]);

  // Storing to db
  string root_folder(argv[1]);
  Datum datum;
  int count = 0;
  const int kMaxKeyLength = 256;
  char key_cstr[kMaxKeyLength];
  int data_size;
  bool data_size_initialized = false;
  int label2 = 0;

  for (int line_id = 0; line_id < lines.size(); ++line_id) {
	  if(!label_inte_bit[lines[line_id].second])
		  continue;

	  if(!label_remap)
		  label2 = lines[line_id].second;
	  else
		  label2 = label_inte_remap[lines[line_id].second];
	  out_listfile<<lines[line_id].first<<" "<<label2<<std::endl;

	  bool status;
	  if(resize_short_side > 0 || resize_long_side > 0){
      status = ReadImageToDatumShortLongSide(root_folder + lines[line_id].first,
          label2, resize_short_side, resize_long_side, is_color, &datum);
	  } else {
	  	status = ReadImageToDatum(root_folder + lines[line_id].first,
				  label2, resize_height, resize_width, is_color, &datum);
	  }

	  if (!status) {
      continue;
    }
	  if(check_size){
	    if (!data_size_initialized) {
	      data_size = datum.channels() * datum.height() * datum.width();
	      data_size_initialized = true;
	    } else {
	      const string& data = datum.data();
	      CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
	          << data.size();
	    }
	  }

    // sequential
    snprintf(key_cstr, kMaxKeyLength, "%08d_%s", line_id,
        lines[line_id].first.c_str());
    string value;
    datum.SerializeToString(&value);
    string keystr(key_cstr);

    // Put in db
    txn->Put(keystr, value);

//    // Put in db
//    if (db_backend == "leveldb") {  // leveldb
//      batch->Put(keystr, value);
//    } else if (db_backend == "lmdb") {  // lmdb
//      mdb_data.mv_size = value.size();
//      mdb_data.mv_data = reinterpret_cast<void*>(&value[0]);
//      mdb_key.mv_size = keystr.size();
//      mdb_key.mv_data = reinterpret_cast<void*>(&keystr[0]);
//      CHECK_EQ(mdb_put(mdb_txn, mdb_dbi, &mdb_key, &mdb_data, 0), MDB_SUCCESS)
//          << "mdb_put failed";
//    } else {
//      LOG(FATAL) << "Unknown db backend " << db_backend;
//    }

    if (++count % 1000 == 0) {
      // Commit db
      txn->Commit();
      txn.reset(db->NewTransaction());

//      // Commit txn
//      if (db_backend == "leveldb") {  // leveldb
//        db->Write(leveldb::WriteOptions(), batch);
//        delete batch;
//        batch = new leveldb::WriteBatch();
//      } else if (db_backend == "lmdb") {  // lmdb
//        CHECK_EQ(mdb_txn_commit(mdb_txn), MDB_SUCCESS)
//            << "mdb_txn_commit failed";
//        CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)
//            << "mdb_txn_begin failed";
//      } else {
//        LOG(FATAL) << "Unknown db backend " << db_backend;
//      }
      LOG(ERROR) << "Processed " << count << " files.";
    }
  }
  out_listfile.close();

  // write the last batch
  if (count % 1000 != 0) {
    txn->Commit();
    LOG(ERROR) << "Processed " << count << " files.";

//    if (db_backend == "leveldb") {  // leveldb
//      db->Write(leveldb::WriteOptions(), batch);
//      delete batch;
//      delete db;
//    } else if (db_backend == "lmdb") {  // lmdb
//      CHECK_EQ(mdb_txn_commit(mdb_txn), MDB_SUCCESS) << "mdb_txn_commit failed";
//      mdb_close(mdb_env, mdb_dbi);
//      mdb_env_close(mdb_env);
//    } else {
//      LOG(FATAL) << "Unknown db backend " << db_backend;
//    }
//    LOG(ERROR) << "Processed " << count << " files.";
  }
  return 0;
}
