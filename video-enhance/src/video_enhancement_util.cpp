#include "video_enhancement_util.hpp"

#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"

#include <algorithm>

using boost::shared_ptr;
using namespace std;

void read_image_names_from_list(const char* image_list_file, std::vector<std::string>& img_names){
	img_names.clear();
	ifstream img_list_f(image_list_file);
	CHECK(img_list_f.is_open()) << " Open file " << image_list_file;
	string line;
	while (getline(img_list_f, line)) {
		img_names.push_back(line);
	}
	img_list_f.close();
}


caffe::Datum read_LAB_image_from_database(string database_path, string image_name){
	shared_ptr<caffe::db::DB> image_db(caffe::db::GetDB(string("lmdb")));
	image_db->Open(database_path, caffe::db::READ);
	shared_ptr<caffe::db::Transaction> image_txn(image_db->NewTransaction(true));
	string stringdata=image_txn->GetValue(image_name);
	caffe::Datum datum;
	datum.ParseFromString(stringdata);
	image_db->Close();
	return datum;
}

void read_in_LAB_images_from_database(string database_path,
		const vector<string> &img_names, map<string, caffe::Datum>& themap) {
	/*
	 * */
	for(int i = 0;i < img_names.size();++i) {
		caffe::Datum datum = read_LAB_image_from_database(database_path, img_names[i]);
		themap[img_names[i]]=datum;
	}
}

void get_segment_median_center(vecInt* segment, int img_width, int img_height,
		int &seg_x, int &seg_y){
	vector<int> px,py;
	for(int i=0;i<segment->size(); ++i) {
		int index = (*segment)[i];
		int y = index / img_width;
		int x = index % img_width;
		px.push_back(x);
		py.push_back(y);
	}
	std::sort(px.begin(), px.end());
	std::sort(py.begin(), py.end());
	seg_x = px[px.size()/2];
	seg_y = py[py.size()/2];
}
