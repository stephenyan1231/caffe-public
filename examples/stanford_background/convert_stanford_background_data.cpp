#include <fstream>  // NOLINT(readability/streams)
#include <algorithm>
#include <string>
#include <vector>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

using namespace caffe;
using boost::scoped_ptr;

DEFINE_bool(shuffle, false, "Randomly shuffle the order of images") ;
DEFINE_bool(encoded, false,
		"When this option is on, the encoded image will be save in datum") ;
DEFINE_int32(min_height, 1, "minimal image height") ;
DEFINE_int32(min_width, 1, "minimal image width") ;
DEFINE_string(backend, "lmdb",
		"The backend {lmdb, leveldb} for storing the result") ;

int main(int argc, char** argv) {
	::google::InitGoogleLogging(argv[0]);

	gflags::SetUsageMessage(
			"Convert a list of images and their pixel annotation into a leveldb/lmdb \n"
					"format used as input for Caffe.\n"
					"Usage:\n"
					"    convert_stanford_background_data [FLAGS] IMAGE_DIR/ LABEL_DIR/ LIST_FILE DB_NAME\n");

	gflags::ParseCommandLineFlags(&argc, &argv, true);

	if (argc < 5) {
		gflags::ShowUsageWithFlagsRestrict(argv[0],
				"examples/stanford_background/convert_stanford_background_data");
		return 1;
	}

	LOG(INFO)<<"min height "<<FLAGS_min_height<<" min width "<<FLAGS_min_width;

	std::string img_name;
	std::vector<std::string> img_names;
	std::ifstream img_list_file(argv[3]);
	while (img_list_file >> img_name) {
		img_names.push_back(img_name);
	}
	img_list_file.close();
	LOG(INFO)<<"read "<<img_names.size()<<" images from the list";

	if (FLAGS_shuffle) {
		// randomly shuffle data
		LOG(INFO)<< "Shuffling data";
		shuffle(img_names.begin(), img_names.end());
	}

	// Create new DB
	scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
	db->Open(argv[4], db::NEW);
	scoped_ptr<db::Transaction> txn(db->NewTransaction());

	// Storing to db
	std::string img_dir(argv[1]);
	std::string label_dir(argv[2]);
	int count = 0;
	const int kMaxKeyLength = 256;
	char key_cstr[kMaxKeyLength];

	for (int i = 0; i < img_names.size(); ++i) {
		SemanticLabelingDatum datum;
		bool status = ReadImageToSemanticLabelingDatum(
				img_dir + img_names[i] + string(".jpg"),
				label_dir + img_names[i] + string(".regions.txt"), FLAGS_min_height,
				FLAGS_min_width, &datum);
		if (!status) {
			continue;
		}
		// database item key
		int length = snprintf(key_cstr, kMaxKeyLength, "%s", img_names[i].c_str());

		string out;
		CHECK(datum.SerializeToString(&out));
		txn->Put(string(key_cstr, length), out);

		if (++count % 100 == 0) {
			txn->Commit();
			txn.reset(db->NewTransaction());
			LOG(ERROR)<<"Processed "<<count<<" images";
		}
	}
	if (count % 100 != 0) {
		txn->Commit();
		LOG(ERROR)<<"Processed "<<count<<" images";
	}
	return 0;
}
