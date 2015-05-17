// Copyright 2015 Zhicheng Yan

#include <glog/logging.h>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

#include <string>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <sstream>      // std::stringstream
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <boost/algorithm/string.hpp>

#include "segment/image.h"
#include "segment/misc.h"
#include "segment/pnmfile.h"
#include "segment/segment-image.h"

#include "video_enhancement_util.hpp"

using namespace std;

using boost::shared_ptr;
namespace db = caffe::db;

const int SEMANTIC_CONTEXT_FTR_DIM = 4096;
const int CV_MAT_CHANNELS = 4;
const int COMMIT_FREQUENCY = 1000;

DEFINE_string(train_image_list, "", "The training image list file") ;

DEFINE_string(original_image_ppm_dir, "",
		"The PPM image folder of original images") ;
DEFINE_string(original_image_LAB_lmdb, "",
		"The lmdb database of original CIELAB image") ;
DEFINE_string(enhanced_image_LAB_lmdb, "",
		"The lmdb database of enhanced CIELAB image") ;

// graph-cut segmentation parameters
DEFINE_double(graphcut_sigma, 0.25, "Graph cut segmentation parameter \sigma") ;
DEFINE_double(graphcut_k, 3, "Graph cut segmentation parameter k") ;
DEFINE_int32(graphcut_min_size, 10, "Graph cut segmentation segment min size") ;
DEFINE_int32(graphcut_max_size, 20, "Graph cut segmentation segment max size") ;

DEFINE_int32(segment_random_sample_num, 10,
		"number of pixels randomly sampled within a segment") ;

DEFINE_bool(compute_global_feature, false, "Compute image global features?");
DEFINE_string(global_feature_lmdb_path, "",
		"LMDB database path where precomputed global feature resides");

DEFINE_bool(compute_semantic_context_feature, false,
		"Compute image local semantic context features on the fly?");
DEFINE_string(semantic_context_feature_binary_dirs, "",
		"the folders (possibly more than one) where precomputed semantic context feature binary file resides");

DEFINE_string(out_training_segment_lmdb, "",
		"the output training lmdb database");

typedef struct {
	int h;
	int w;
} ImageSize;

std::map<std::string, caffe::Datum> in_LAB_images;
std::map<std::string, caffe::Datum> out_LAB_images;

typedef std::pair<int, int> ImgID_SegID;

int main(int argc, char** argv) {
	FLAGS_alsologtostderr = 1;
	// Usage message.
	gflags::SetUsageMessage("command line brew\n"
			"usage: caffe <command> <args>\n\n"
			"commands:\n"
			"  train           train or finetune a model\n"
			"  test            score a model\n"
			"  device_query    show GPU diagnostic information\n"
			"  time            benchmark model execution time");
	// Run tool or show usage.
	caffe::GlobalInit(&argc, &argv);

	std::vector<std::string> train_img_names;
	read_image_names_from_list(FLAGS_train_image_list.c_str(), train_img_names);
	LOG(INFO)<<"Use "<<train_img_names.size()<<" training images";

	LOG(INFO)<<"Use graph-cut to segment images";
	vector<ImageSize> image_sizes(train_img_names.size());
	vector<map<int, vecInt*> > image_segments;
	int total_segments = 0;
	for (int i = 0; i < train_img_names.size(); ++i) {
		string ppm_img_path = FLAGS_original_image_ppm_dir + train_img_names[i]
				+ string(".ppm");
		LOG(INFO)<<"read ppm image "<<ppm_img_path;
		image<rgb> *input = loadPPM(ppm_img_path.c_str());
		map<int, vecInt*> comps;
		int num_ccs;
		image<rgb> *seg = segment_image(input, FLAGS_graphcut_sigma,
				FLAGS_graphcut_k, FLAGS_graphcut_min_size, FLAGS_graphcut_max_size,
				&num_ccs, comps);
		image_sizes[i].h = input->height();
		image_sizes[i].w = input->width();
		delete input;
		delete seg;
		LOG(INFO)<<comps.size()<<" segments";
		image_segments.push_back(comps);
		total_segments += comps.size();
	}

	/*
	 * read image into CIELAB space
	 * */
	read_in_LAB_images_from_database(FLAGS_original_image_LAB_lmdb,
			train_img_names, in_LAB_images);
	read_in_LAB_images_from_database(FLAGS_enhanced_image_LAB_lmdb,
			train_img_names, out_LAB_images);

	/*
	 * compute global features and store them in a lmdb database
	 * */
	std::map<std::string, caffe::Datum> global_ftrs;
	if (FLAGS_compute_global_feature) {
		NOT_IMPLEMENTED;
	} else {
//		LOG(INFO)<<"LMDB database path : "<<FLAGS_global_feature_lmdb_path;
//		shared_ptr<db::DB> global_ftr_db(db::GetDB(string("lmdb")));
//		global_ftr_db->Open(FLAGS_global_feature_lmdb_path, db::READ);
//		shared_ptr<db::Transaction> global_ftr_txn(global_ftr_db->NewTransaction(true));
//
//		for(int i=0;i<train_img_names.size();++i) {
//			LOG(INFO)<<"database key: "<<train_img_names[i];
//			string stringdata=global_ftr_txn->GetValue(train_img_names[i]);
//			caffe::Datum datum;
//			datum.ParseFromString(stringdata);
//			LOG(INFO)<<"datum channels "<<datum.channels();
//			global_ftrs[train_img_names[i]] = datum;
//		}
//		global_ftr_db->Close();
	}

	/*
	 * each training segment will be saved into a training lmdb database
	 * first generate a random key for each segment in each training images
	 * */
	vector<int> segment_key(total_segments);
	for (int i = 0; i < total_segments; ++i) {
		segment_key[i] = i;
	}
	random_shuffle(segment_key.begin(), segment_key.end());
//	LOG(INFO)<<"segment keys "<<segment_key[0]<<" "<<segment_key[1]<<" "<<segment_key[2];

	int segment_count = 0;
	map<int, ImgID_SegID> key2ImgID_SegID;
	vector<vector<int> > imgID2Key(train_img_names.size());
	for (int i = 0; i < train_img_names.size(); ++i) {
		for (map<int, vecInt*>::iterator it = image_segments[i].begin();
				it != image_segments[i].end(); ++it) {
			key2ImgID_SegID[segment_key[segment_count]] = std::make_pair(i,
					it->first);
			imgID2Key[i].push_back(segment_key[segment_count]);
			segment_count++;
		}
	}

	/*
	 * either compute or load semantic local context feature
	 * Questions: consider various ways to normalize semantic context features
	 * */LOG(INFO)<<"output LMDB database path : "<<FLAGS_out_training_segment_lmdb;
	shared_ptr<db::DB> out_training_segment_db(db::GetDB(string("lmdb")));
	out_training_segment_db->Open(FLAGS_out_training_segment_lmdb, db::NEW);
	shared_ptr<db::Transaction> out_training_segment_txn(
			out_training_segment_db->NewTransaction(false));

	const int kMaxKeyLength = 256;
	char key_cstr[kMaxKeyLength];

	if (FLAGS_compute_semantic_context_feature) {
		NOT_IMPLEMENTED;
	} else {
		LOG(INFO)<<"Load precomputed semantic context feature binary files from folder: "
		<<FLAGS_semantic_context_feature_binary_dirs;

		std::vector<std::string> semantic_context_feature_binary_dirs;
		boost::split(semantic_context_feature_binary_dirs, FLAGS_semantic_context_feature_binary_dirs,
				boost::is_any_of(","));
		for(int i =0;i<semantic_context_feature_binary_dirs.size();++i) {
			LOG(INFO)<<"semantic_context_feature_binary_dir : "<<semantic_context_feature_binary_dirs[i];
		}

		segment_count = 0;
		for(int i = 0;i < train_img_names.size(); ++i) {
			char fn[32];

			/* load multi-scale semantic context feature
			 * upsample it to match the original image size
			 * */
			vector<vector<cv::Mat*> > context_features;
			for(int j = 0; j < semantic_context_feature_binary_dirs.size(); ++j) {
				sprintf(fn,"%d.txt",i);
				string sem_context_ftr_meta_path = semantic_context_feature_binary_dirs[j] +
				string(fn);
				ifstream sem_context_ftr_meta(sem_context_ftr_meta_path.c_str());
				CHECK(sem_context_ftr_meta.is_open())<<" Open meta file "<<sem_context_ftr_meta_path;
				stringstream ss;
				string line;
				int channels, height, width;
				getline(sem_context_ftr_meta,line);
				ss.clear();
				ss.str(line);
				ss>>channels;
				CHECK(SEMANTIC_CONTEXT_FTR_DIM == channels);
				getline(sem_context_ftr_meta,line);
				ss.clear();
				ss.str(line);
				ss>>height;
				getline(sem_context_ftr_meta,line);
				ss.clear();
				ss.str(line);
				ss>>width;
				sem_context_ftr_meta.close();
				LOG(INFO)<<"meta file:"<<sem_context_ftr_meta_path<<" ch:"<<channels<<" h:"<<height<<" w:"<<width;

				float *sem_context_ftr = new float[channels * height * width];

				sprintf(fn,"%d.dat",i);
				string sem_context_ftr_data_path = semantic_context_feature_binary_dirs[j] + string(fn);
				LOG(INFO)<<"read binary data file: "<<sem_context_ftr_data_path;

				ifstream sem_context_ftr_data(sem_context_ftr_data_path.c_str(), ios::in|ios::binary| ios::ate);
				size_t filesize = sem_context_ftr_data.tellg();
				CHECK_EQ(filesize, sizeof(float) * channels * height * width);
				sem_context_ftr_data.seekg(0, ios::beg);

				CHECK(sem_context_ftr);
				sem_context_ftr_data.read(reinterpret_cast<char*>(sem_context_ftr), sizeof(float)*channels*height*width);
				sem_context_ftr_data.close();

				typedef cv::Vec<float, SEMANTIC_CONTEXT_FTR_DIM> VecChf;
				CHECK_EQ(SEMANTIC_CONTEXT_FTR_DIM % CV_MAT_CHANNELS, 0);
				int num_mat = SEMANTIC_CONTEXT_FTR_DIM / CV_MAT_CHANNELS;

				vector<cv::Mat*> mats(num_mat);
				for(int k = 0;k < num_mat; ++k) {
					mats[k]=new cv::Mat(height, width, CV_32FC4);
					CHECK(mats[k]->channels() == CV_MAT_CHANNELS);
					for(int c=0;c<CV_MAT_CHANNELS;++c) {
						for(int h=0;h<height;++h) {
							for(int w=0;w<width;++w) {
								int index=((k*CV_MAT_CHANNELS+c)*height+h)*width+w;
								mats[k]->at<cv::Vec4f>(h,w)[c]=sem_context_ftr[index];
							}
						}
					}
				}

				delete[] sem_context_ftr;

				LOG(INFO)<<"resize semantic context feature map to original image size:"
				<<image_sizes[i].h<<" "<<image_sizes[i].w;
				vector<cv::Mat*> resampled_mats(num_mat);
				for(int k = 0;k < num_mat; ++k) {
					resampled_mats[k]=new cv::Mat();
					cv::resize(*(mats[k]), *(resampled_mats[k]), cv::Size(image_sizes[i].w,image_sizes[i].h));
					delete mats[k];
				}
				context_features.push_back(resampled_mats);
			}

			/*
			 * assemble one training sample per image segment
			 * */
			caffe::Datum &in_img = in_LAB_images[train_img_names[i]];
			caffe::Datum &out_img = out_LAB_images[train_img_names[i]];
			CHECK_EQ(in_img.channels(), out_img.channels());
			CHECK_EQ(in_img.height(), out_img.height());
			CHECK_EQ(in_img.width(), out_img.width());

//			vector<int> px,py;
			vector<int> indices;
			for(int j=0;j<image_segments[i].size();++j) {
				caffe::ImageEnhancementDatum train_segment;
				train_segment.set_image_name(train_img_names[i]);
				int key = imgID2Key[i][j];
				CHECK(i == key2ImgID_SegID[imgID2Key[i][j]].first);
				int segID = key2ImgID_SegID[imgID2Key[i][j]].second;
//				px.clear();
//				py.clear();
				indices.clear();
				for(int k=0;k<image_segments[i][segID]->size(); ++k) {
					int index = (*image_segments[i][segID])[k];
//					int y = index / in_img.width();
//					int x = index % in_img.width();
//					px.push_back(x);
//					py.push_back(y);
					indices.push_back(index);
				}
//				std::sort(px.begin(), px.end());
//				std::sort(py.begin(), py.end());

				// take median value to localize segment center
				int seg_x = 0, seg_y = 0;
				get_segment_median_center(image_segments[i][segID], in_img.width(),
						in_img.height(), seg_x, seg_y);
//				int seg_x = px[px.size()/2];
//				int seg_y= py[py.size()/2];
				for(int k=0;k<in_img.channels();++k) {
					train_segment.add_pixel_ftr(
							in_img.float_data((k*in_img.height()+seg_y)*in_img.width() + seg_x));
				}

				// semantic context feature
				for(int k = 0; k < semantic_context_feature_binary_dirs.size(); ++k) {
					for(int g = 0; g < (SEMANTIC_CONTEXT_FTR_DIM / CV_MAT_CHANNELS); ++g) {
						for(int c=0;c<CV_MAT_CHANNELS;++c) {
							/*
							 * directly append multi-scale semantic feature w/o any normalization
							 */
							train_segment.add_semantic_context_ftr(context_features[k][g]->at<cv::Vec4f>(seg_y,seg_x)[c]);
						}
					}
				}

				// randomly sample a few pixels within the segment
				std::random_shuffle(indices.begin(), indices.end());
//				LOG(INFO)<<"shuffled indices "<<indices[0]<<" "<<indices[1]<<" "<<indices[2];
				for(int k=0;k<FLAGS_segment_random_sample_num;++k) {
					int index = indices[k % indices.size()];
					int y = index / in_img.width();
					int x = index % in_img.width();
					for(int c=0;c<in_img.channels();++c) {
						int datum_index = (c*in_img.height()+y)*in_img.width()+x;
						train_segment.add_original_lab_color(in_img.float_data(datum_index));
						train_segment.add_enhanced_lab_color(out_img.float_data(datum_index));
					}
				}

				int length = snprintf(key_cstr, kMaxKeyLength, "%09d", key);
				string stringdata;
				CHECK(train_segment.SerializeToString(&stringdata));
				out_training_segment_txn->Put(string(key_cstr, length), stringdata);
				segment_count++;
				if(segment_count % COMMIT_FREQUENCY == 0) {
					out_training_segment_txn->Commit();
					out_training_segment_txn.reset(out_training_segment_db->NewTransaction(false));
					LOG(ERROR) << "Processed "<<segment_count<<" segments";
				}
			}

			/*clear work*/
			for(int j=0;j<context_features.size();++j) {
				for(int k=0;k<context_features[j].size();++k) {
					delete context_features[j][k];
				}
				context_features[j].clear();
			}
			context_features.clear();
		}
		if(segment_count % COMMIT_FREQUENCY != 0) {
			out_training_segment_txn->Commit();
			LOG(ERROR) << "Processed "<<segment_count<<" segments";
		}
		out_training_segment_db->Close();
	}

	/*clean work*/
	for (int i = 0; i < train_img_names.size(); ++i) {
		for (map<int, vecInt*>::iterator it = image_segments[i].begin();
				it != image_segments[i].end(); ++it) {
			delete it->second;
		}
		image_segments[i].clear();
	}
	image_segments.clear();

	return 0;
}
