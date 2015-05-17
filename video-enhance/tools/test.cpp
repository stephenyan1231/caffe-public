// Copyright 2015 Zhicheng Yan

#include <glog/logging.h>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/net.hpp"

#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>      // std::stringstream
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <boost/algorithm/string.hpp>

#include "segment/image.h"
#include "segment/misc.h"
#include "segment/pnmfile.h"
#include "segment/segment-image.h"

#include "video_enhancement_util.hpp"

using namespace std;
using namespace caffe;

using caffe::Net;

using boost::shared_ptr;

// testing options

DEFINE_string(test_image_list, "", "The testing image list file") ;
DEFINE_string(original_image_ppm_dir, "", "The original 2PPM image folder") ;
DEFINE_string(original_image_LAB_lmdb, "",
		"The lmdb database of original CIELAB image") ;
DEFINE_string(enhanced_image_LAB_lmdb, "",
		"The lmdb database of enhanced CIELAB image") ;
DEFINE_string(global_feature_lmdb_path, "",
		"LMDB database path where precomputed global feature resides") ;

DEFINE_string(semantic_context_feature_model, "",
		"The definition file of the net for extracting semantic context feature") ;
DEFINE_string(semantic_context_feature_weights, "",
		"The model weights file (.caffemodel) of the net for extracting semantic context feature") ;
DEFINE_string(semantic_context_feature_scales, "1024,2048",
		"The scales (length of long edge) used for semantic context feature extraction") ;

DEFINE_string(video_enhance_model, "",
		"The definition file of video enhancement testing net") ;
DEFINE_string(video_enhance_weights, "",
		"The model weights file (.caffemodel) of video enhancement testing net") ;

// graph-cut segmentation parameters
DEFINE_double(graphcut_sigma, 0.25, "Graph cut segmentation parameter \sigma") ;
DEFINE_double(graphcut_k, 3, "Graph cut segmentation parameter k") ;
DEFINE_int32(graphcut_min_size, 10, "Graph cut segmentation segment min size") ;
DEFINE_int32(graphcut_max_size, 20, "Graph cut segmentation segment max size") ;

DEFINE_string(gpu, "0", "GPU IDs") ;

DEFINE_string(semantic_context_feature_extraction_mean_values, "104,117,123",
		"image data preprocessing before extracting semantic context feature") ;
DEFINE_string(semantic_context_feature_extraction_layer_name, "stitch_fc7",
		"the name of the layer whose output feature map is used as semantic context feature") ;

DEFINE_string(global_feature_mean_file, "",
		"the file path storing the mean value of global feature") ;
DEFINE_string(semantic_context_feature_mean_file, "",
		"the file path storing the mean value of semantic context feature") ;
DEFINE_string(pixel_feature_mean_file, "",
		"the file path storing the mean value of pixel-level feature") ;
DEFINE_double(global_feature_scaling, 50.0, "scaling factor of global feature") ;
DEFINE_double(semantic_context_feature_scaling, 0.5,
		"scaling factor of semantic context feature") ;
DEFINE_double(pixel_feature_scaling, 0.5,
		"scaling factor of pixel-level feature") ;

DEFINE_string(output_image_dir, "",
		"the folder for saving the enhanced images") ;

const int SEMANTIC_CONTEXT_FTR_DIM = 4096;
const int PIXEL_FTR_DIM = 3;

const int VIDEO_ENHANCEMENT_NET_BATCHSIZE = 200;
const int QUAD_COLOR_BASIS_DIM = 10;

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

	std::vector<int> device_ids;
	if (FLAGS_gpu.length() > 0) {
		LOG(INFO)<< "Use GPU with device ID " << FLAGS_gpu;
		Caffe::set_mode(Caffe::GPU);
		device_ids = caffe::parse_int_list(FLAGS_gpu);
		CHECK_EQ(device_ids.size(), 1) << "Allow to use only a single GPU";
	} else {
		LOG(INFO) << "Use CPU.";
		Caffe::set_mode(Caffe::CPU);
	}

	// Initialize Caffe.
	Caffe::set_phase(Caffe::TEST);
	Caffe::InitDevices(device_ids);

	// extract semantic context feature at multiple scales
	vector<int> semantic_context_feature_scales;
	vector<string> semantic_context_feature_scales_strings;
	boost::split(semantic_context_feature_scales_strings,
			FLAGS_semantic_context_feature_scales, boost::is_any_of(","));
	for (int i = 0; i < semantic_context_feature_scales_strings.size(); ++i) {
		semantic_context_feature_scales.push_back(
				atoi(semantic_context_feature_scales_strings[i].c_str()));
		LOG(INFO)<< "semantic context feature extraction scale: "
		<< semantic_context_feature_scales[i];
	}

	shared_ptr<Net<float> > semantic_context_feature_net(
			new Net<float>(FLAGS_semantic_context_feature_model));
	semantic_context_feature_net->CopyTrainedLayersFrom(
			FLAGS_semantic_context_feature_weights);

	std::vector<std::string> test_img_names;
	read_image_names_from_list(FLAGS_test_image_list.c_str(), test_img_names);
	LOG(INFO)<< "Use " << test_img_names.size() << " testing images";

	LOG(INFO)<< "LMDB database path : " << FLAGS_global_feature_lmdb_path;
	shared_ptr<caffe::db::DB> global_ftr_db(caffe::db::GetDB(string("lmdb")));
	global_ftr_db->Open(FLAGS_global_feature_lmdb_path, caffe::db::READ);
	shared_ptr<caffe::db::Transaction> global_ftr_txn(
			global_ftr_db->NewTransaction(true));

	vector<int> mean_values(3);
	vector<string> mean_values_strings;
	boost::split(mean_values_strings,
			FLAGS_semantic_context_feature_extraction_mean_values,
			boost::is_any_of(","));
	CHECK_EQ(mean_values_strings.size(), 3);
	for (int i = 0; i < 3; ++i) {
		mean_values[i] = atoi(mean_values_strings[i].c_str());
		LOG(INFO)<< "Image preprocessing. mean value " << i << ":"
		<< mean_values[i];
		CHECK_LE(mean_values[i], 255);
	}

	LOG(INFO)<< "Video enhancement net model file: "
	<< FLAGS_video_enhance_model;
	shared_ptr<Net<float> > video_enhancement_net(
			new Net<float>(FLAGS_video_enhance_model));
	video_enhancement_net->CopyTrainedLayersFrom(FLAGS_video_enhance_weights);

	// load mean values of the inputs to video enhancement net
	BlobProto global_ftr_mean_blob_proto;
	ReadProtoFromBinaryFileOrDie(FLAGS_global_feature_mean_file.c_str(),
			&global_ftr_mean_blob_proto);
	Blob<float> global_ftr_mean;
	global_ftr_mean.FromProto(global_ftr_mean_blob_proto);
	const float *global_ftr_mean_data = global_ftr_mean.cpu_data();

	BlobProto semantic_context_ftr_mean_blob_proto;
	ReadProtoFromBinaryFileOrDie(
			FLAGS_semantic_context_feature_mean_file.c_str(),
			&semantic_context_ftr_mean_blob_proto);
	Blob<float> semantic_context_ftr_mean;
	semantic_context_ftr_mean.FromProto(semantic_context_ftr_mean_blob_proto);
	CHECK_EQ(semantic_context_ftr_mean.channels(),
			SEMANTIC_CONTEXT_FTR_DIM * semantic_context_feature_scales.size());
	const float *semantic_context_ftr_mean_data =
			semantic_context_ftr_mean.cpu_data();

	BlobProto pixel_ftr_mean_blob_proto;
	ReadProtoFromBinaryFileOrDie(FLAGS_pixel_feature_mean_file.c_str(),
			&pixel_ftr_mean_blob_proto);
	Blob<float> pixel_ftr_mean;
	pixel_ftr_mean.FromProto(pixel_ftr_mean_blob_proto);
	const float *pixel_ftr_mean_data = pixel_ftr_mean.cpu_data();

	float testing_error_sum =0;
	vector<float> testing_error(test_img_names.size());

	// start testing
	for (int i = 0; i < test_img_names.size(); ++i) {
		string ppm_img_path = FLAGS_original_image_ppm_dir + test_img_names[i]
				+ string(".ppm");
		LOG(INFO)<<"--testing case "<<i<<" out of "<<test_img_names.size()<<"--";
		LOG(INFO)<<"Read ppm image " << ppm_img_path;
		image<rgb> *input = loadPPM(ppm_img_path.c_str());
		map<int, vecInt*> comps;
		int num_ccs;
		image<rgb> *seg = segment_image(input, FLAGS_graphcut_sigma,
				FLAGS_graphcut_k, FLAGS_graphcut_min_size,
				FLAGS_graphcut_max_size, &num_ccs, comps);
		int img_height = input->height();
		int img_width = input->width();
		LOG(INFO)<< "original image height " << img_height << " width "
		<< img_width;

		delete seg;
		LOG(INFO)<< comps.size() << " segments";
		map<int, pair<int, int> > seg_centers;
		for (map<int, vecInt*>::iterator it = comps.begin(); it != comps.end();
				++it) {
			seg_centers[it->first] = make_pair(0, 0);
			get_segment_median_center(it->second, img_width, img_height,
					seg_centers[it->first].first,
					seg_centers[it->first].second);
			DLOG(INFO)<< "segment cetner " << seg_centers[it->first].first
			<< " " << seg_centers[it->first].second;
		}

		caffe::Datum original_img_datum = read_LAB_image_from_database(
				FLAGS_original_image_LAB_lmdb, test_img_names[i]);
		caffe::Datum enhanced_img_datum = read_LAB_image_from_database(
				FLAGS_enhanced_image_LAB_lmdb, test_img_names[i]);
		CHECK_EQ(img_height, original_img_datum.height());
		CHECK_EQ(img_width, original_img_datum.width());

		CHECK_EQ(enhanced_img_datum.channels(), original_img_datum.channels());
		CHECK_EQ(enhanced_img_datum.channels(), 3);
		CHECK_EQ(enhanced_img_datum.height(), original_img_datum.height());
		CHECK_EQ(enhanced_img_datum.width(), original_img_datum.width());

		//Global feature should be computed on-the-fly
		//Hack! get global feature from a pre-computed database
		string stringdata = global_ftr_txn->GetValue(test_img_names[i]);
		caffe::Datum global_feature_datum;
		global_feature_datum.ParseFromString(stringdata);
		int GLOBAL_FTR_DIM = global_feature_datum.channels();
		LOG(INFO)<< "global feature dim:" << global_feature_datum.channels();

		// segment-wise semantic context feature, pixel-level feature
		Blob<float> semantic_context_ftr_seg_blob;
		semantic_context_ftr_seg_blob.Reshape(comps.size(),
				semantic_context_feature_scales.size()
						* SEMANTIC_CONTEXT_FTR_DIM, 1, 1);

		// extract semantic context feature
		const vector<string>& blob_names =
				semantic_context_feature_net->blob_names();
		const vector<int>& input_blob_indices =
				semantic_context_feature_net->input_blob_indices();
		int resize_height = 0, resize_width = 0;

		cv::Mat* cv_img = new cv::Mat(img_height, img_width, CV_8UC3);
		for (int h = 0; h < img_height; ++h) {
			for (int w = 0; w < img_width; ++w) {
				cv_img->at<cv::Vec3b>(h, w)[0] = imRef(input,w,h).b;
				cv_img->at<cv::Vec3b>(h,w)[1]=imRef(input,w,h).g;
				cv_img->at<cv::Vec3b>(h,w)[2]=imRef(input,w,h).r;
			}
		}

		for (int j = 0; j < semantic_context_feature_scales.size(); ++j) {
			shared_ptr<Blob<float> > data_blob =
					semantic_context_feature_net->blob_by_name("data", 0);
			if (img_height > img_width) {
				resize_height = semantic_context_feature_scales[j];
				resize_width = ceil(
						(float) resize_height * (float) img_width
								/ (float) img_height);
			} else {
				resize_width = semantic_context_feature_scales[j];
				resize_height = ceil(
						(float) resize_width * (float) img_height
								/ (float) img_width);
			}
			LOG(INFO)<< "resize height " << resize_height << " width "
			<< resize_width;
			cv::Mat cv_resized_img;
			cv::resize(*cv_img, cv_resized_img,
					cv::Size(resize_width, resize_height));
			// fill the data blob, run the forward pass and extract layer blob feature
			data_blob->Reshape(1, 3, resize_height, resize_width);
			float* data_blob_data = data_blob->mutable_cpu_data();
			for (int h = 0; h < resize_height; ++h) {
				const uchar* ptr = cv_resized_img.ptr<uchar>(h);
				int img_index = 0;
				for (int w = 0; w < resize_width; ++w) {
					for (int c = 0; c < 3; ++c) {
						data_blob_data[data_blob->offset(0, c, h, w)] =
								static_cast<float>(ptr[img_index++])
										- mean_values[c];
					}
				}
			}
			NetThread<float> *net_thread =
					semantic_context_feature_net->GetNetThreads()[0];
			const vector<shared_ptr<Layer<float> > >& net_layers =
					net_thread->layers();
			semantic_context_feature_net->ForwardFromTo(0,
					net_layers.size() - 1);
			shared_ptr<Blob<float> > semantic_ftr_blob =
					semantic_context_feature_net->blob_by_name(
							FLAGS_semantic_context_feature_extraction_layer_name,
							0);
			CHECK_EQ(semantic_ftr_blob->num(), 1);
			LOG(INFO)<< "semantic feature blob shape:"
			<< semantic_ftr_blob->num() << " "
			<< semantic_ftr_blob->channels() << " "
			<< semantic_ftr_blob->height() << " "
			<< semantic_ftr_blob->width();
			const float* semantic_ftr_blob_data = semantic_ftr_blob->cpu_data();
			float *semantic_context_ftr_seg_blob_data =
					semantic_context_ftr_seg_blob.mutable_cpu_data();
			int p = 0;
			for (map<int, vecInt*>::iterator it = comps.begin();
					it != comps.end(); ++it) {
				int seg_ctr_x = seg_centers[it->first].first;
				int seg_ctr_y = seg_centers[it->first].second;
				int NN_seg_ctr_x = floor(
						((float) seg_ctr_x / (float) img_width)
								* semantic_ftr_blob->width());
				int NN_seg_ctr_y = floor(
						((float) seg_ctr_y / (float) img_height)
								* semantic_ftr_blob->height());
				DLOG(INFO)<< "NN_seg_ctr_x " << NN_seg_ctr_x
				<< " NN_seg_ctr_y " << NN_seg_ctr_y;
				int offset = semantic_context_ftr_seg_blob.offset(p,
						j * SEMANTIC_CONTEXT_FTR_DIM, 0, 0);
				for (int k = 0; k < SEMANTIC_CONTEXT_FTR_DIM; ++k) {
					semantic_context_ftr_seg_blob_data[offset + k] =
							semantic_ftr_blob_data[semantic_ftr_blob->offset(0,
									k, NN_seg_ctr_y, NN_seg_ctr_x)];
				}
				p++;
			}

		}

		/*
		 * prepare testing batch for video enhancement net
		 * */
		int num_batch = (comps.size() + VIDEO_ENHANCEMENT_NET_BATCHSIZE - 1)
				/ VIDEO_ENHANCEMENT_NET_BATCHSIZE;
		map<int, vecInt*>::iterator seg_iter = comps.begin();

		shared_ptr<Blob<float> > global_ftr_blob =
				video_enhancement_net->blob_by_name("global_ftr", 0);
		shared_ptr<Blob<float> > semantic_context_ftr_blob =
				video_enhancement_net->blob_by_name("semantic_context_ftr", 0);
		shared_ptr<Blob<float> > pixel_ftr_blob =
				video_enhancement_net->blob_by_name("pixel_ftr", 0);

		const float* semantic_context_ftr_seg_blob_data =
				semantic_context_ftr_seg_blob.cpu_data();

		NetThread<float> *video_enhancement_net_thread =
				video_enhancement_net->GetNetThreads()[0];
		const vector<shared_ptr<Layer<float> > >& video_enhancement_net_layers =
				video_enhancement_net_thread->layers();

		cv::Mat pred_LAB_img(img_height, img_width, CV_32FC3);

		Blob<float> color_basis, predicted_color;

		for (int j = 0; j < num_batch; ++j) {
			int start = j * VIDEO_ENHANCEMENT_NET_BATCHSIZE;
			int end = std::min(int(comps.size()),
					start + VIDEO_ENHANCEMENT_NET_BATCHSIZE);
			global_ftr_blob->Reshape(end - start, GLOBAL_FTR_DIM, 1, 1);
			semantic_context_ftr_blob->Reshape(end - start,
					SEMANTIC_CONTEXT_FTR_DIM
							* semantic_context_feature_scales.size(), 1, 1);
			pixel_ftr_blob->Reshape(end - start, PIXEL_FTR_DIM, 1, 1);
			float *global_ftr_blob_data = global_ftr_blob->mutable_cpu_data();
			float *semantic_context_ftr_blob_data =
					semantic_context_ftr_blob->mutable_cpu_data();
			float *pixel_ftr_blob_data = pixel_ftr_blob->mutable_cpu_data();

			vector<int> seg_ids;
			for (int k = 0; k < end - start; ++k) {
				int offset1 = global_ftr_blob->offset(k);
				int offset2 = semantic_context_ftr_blob->offset(k);
				int offset3 = pixel_ftr_blob->offset(k);

				for (int d = 0; d < GLOBAL_FTR_DIM; ++d) {
					global_ftr_blob_data[offset1 + d] =
							FLAGS_global_feature_scaling
									* (global_feature_datum.float_data(d)
											- global_ftr_mean_data[d]);
				}
				int num_scale = semantic_context_feature_scales.size();
				for (int d = 0; d < SEMANTIC_CONTEXT_FTR_DIM * num_scale; ++d) {
					semantic_context_ftr_blob_data[offset2 + d] =
							FLAGS_semantic_context_feature_scaling
									* (semantic_context_ftr_seg_blob_data[semantic_context_ftr_seg_blob.offset(
											start + k, d)]
											- semantic_context_ftr_mean_data[d]);
				}
				for (int d = 0; d < PIXEL_FTR_DIM; ++d) {
					int seg_ctr_x = seg_centers[seg_iter->first].first;
					int seg_ctr_y = seg_centers[seg_iter->first].second;
					int index = (d * img_height + seg_ctr_y) * img_width
							+ seg_ctr_x;
					pixel_ftr_blob_data[offset3 + d] =
							FLAGS_pixel_feature_scaling
									* (original_img_datum.float_data(index)
											- pixel_ftr_mean_data[d]);
				}
				seg_ids.push_back(seg_iter->first);
				seg_iter++;
			}
			video_enhancement_net->ForwardFromTo(0,
					video_enhancement_net_layers.size() - 1);
			shared_ptr<Blob<float> > reglayer_Lab_blob =
					video_enhancement_net->blob_by_name("reglayer_Lab", 0);
			const float* reglayer_Lab_blob_data = reglayer_Lab_blob->cpu_data();

			for (int k = 0; k < end - start; ++k) {
				int num_pix = comps[seg_ids[k]]->size();
				color_basis.Reshape(num_pix, QUAD_COLOR_BASIS_DIM, 1, 1);
				predicted_color.Reshape(num_pix, 3, 1, 1);
				float* color_basis_data = color_basis.mutable_cpu_data();
				float* predicted_color_data =
						predicted_color.mutable_cpu_data();
				for (int p = 0; p < num_pix; ++p) {
					int pix_x = (*comps[seg_ids[k]])[p] % img_width;
					int pix_y = (*comps[seg_ids[k]])[p] / img_width;
					float L = original_img_datum.float_data(
							(*comps[seg_ids[k]])[p]);
					float a = original_img_datum.float_data(
							(*comps[seg_ids[k]])[p] + img_width * img_height);
					float b = original_img_datum.float_data(
							(*comps[seg_ids[k]])[p]
									+ img_width * img_height * 2);
					color_basis_data[color_basis.offset(p)] = L * L;
					color_basis_data[color_basis.offset(p) + 1] = a * a;
					color_basis_data[color_basis.offset(p) + 2] = b * b;
					color_basis_data[color_basis.offset(p) + 3] = L * a;
					color_basis_data[color_basis.offset(p) + 4] = L * b;
					color_basis_data[color_basis.offset(p) + 5] = a * b;
					color_basis_data[color_basis.offset(p) + 6] = L;
					color_basis_data[color_basis.offset(p) + 7] = a;
					color_basis_data[color_basis.offset(p) + 8] = b;
					color_basis_data[color_basis.offset(p) + 9] = 1;
				}
				caffe_cpu_gemm<float>(CblasNoTrans, CblasTrans, num_pix, 3,
						QUAD_COLOR_BASIS_DIM, 1.0f, color_basis_data,
						reglayer_Lab_blob_data + reglayer_Lab_blob->offset(k),
						0.0f, predicted_color_data);

				for (int p = 0; p < num_pix; ++p) {
					int pix_x = (*comps[seg_ids[k]])[p] % img_width;
					int pix_y = (*comps[seg_ids[k]])[p] / img_width;
					pred_LAB_img.at<cv::Vec3f>(pix_y, pix_x)[0] =
							predicted_color_data[predicted_color.offset(p)];
					pred_LAB_img.at<cv::Vec3f>(pix_y, pix_x)[1] =
							predicted_color_data[predicted_color.offset(p) + 1];
					pred_LAB_img.at<cv::Vec3f>(pix_y, pix_x)[2] =
							predicted_color_data[predicted_color.offset(p) + 2];
				}
			}

		}

		int pix_total = img_height*img_width;
		vector<float> pix_error(pix_total);
		float pixel_error_sum = 0.0f;
		for(int pix=0,h=0;h<img_height;++h){
			for(int w=0;w<img_width;++w,++pix){
				float L_diff=pred_LAB_img.at<cv::Vec3f>(h,w)[0]-enhanced_img_datum.float_data(h*img_width+w);
				float a_diff=pred_LAB_img.at<cv::Vec3f>(h,w)[1]-enhanced_img_datum.float_data(pix_total+h*img_width+w);
				float b_diff=pred_LAB_img.at<cv::Vec3f>(h,w)[2]-enhanced_img_datum.float_data(2*pix_total+h*img_width+w);
				pix_error[pix]=sqrt(L_diff*L_diff+a_diff*a_diff+b_diff*b_diff);
				pixel_error_sum += pix_error[pix];
			}
		}
		testing_error[i]= pixel_error_sum / pix_total;
		testing_error_sum += testing_error[i];
		LOG(INFO)<<"Test image "<<test_img_names[i]<<"Lab Error: "<<testing_error[i];

		cv::Mat pred_rgb_image_f32;
		cv::Mat pred_rgb_image_u8(img_height, img_width, CV_8UC3);
		cv::cvtColor(pred_LAB_img, pred_rgb_image_f32, CV_Lab2BGR);

		// convert floating-number image data into unsigned char image data
		for(int h=0;h<img_height;++h){
			for(int w=0;w<img_width;++w){
				CHECK_GE(pred_rgb_image_f32.at<cv::Vec3f>(h,w)[0], 0);
				CHECK_LE(pred_rgb_image_f32.at<cv::Vec3f>(h,w)[0], 1.0);
				CHECK_GE(pred_rgb_image_f32.at<cv::Vec3f>(h,w)[1], 0);
				CHECK_LE(pred_rgb_image_f32.at<cv::Vec3f>(h,w)[1], 1.0);
				CHECK_GE(pred_rgb_image_f32.at<cv::Vec3f>(h,w)[2], 0);
				CHECK_LE(pred_rgb_image_f32.at<cv::Vec3f>(h,w)[2], 1.0);
				pred_rgb_image_u8.at<cv::Vec3b>(h,w)[0]=
						static_cast<unsigned char>(floor(pred_rgb_image_f32.at<cv::Vec3f>(h,w)[0]*255.0));
				pred_rgb_image_u8.at<cv::Vec3b>(h,w)[1]=
						static_cast<unsigned char>(floor(pred_rgb_image_f32.at<cv::Vec3f>(h,w)[1]*255.0));
				pred_rgb_image_u8.at<cv::Vec3b>(h,w)[2]=
						static_cast<unsigned char>(floor(pred_rgb_image_f32.at<cv::Vec3f>(h,w)[2]*255.0));
			}
		}

		string save_path = FLAGS_output_image_dir + test_img_names[i] + string(".png");
		LOG(INFO)<<"save predicted image to "<<save_path;
		imwrite(save_path.c_str(), pred_rgb_image_u8);

		// some cleaning work
		for (map<int, vecInt*>::iterator it = comps.begin(); it != comps.end();
				++it) {
			delete it->second;
		}
		delete cv_img;
		delete input;
	}
	LOG(INFO)<<"-----------------------";
	LOG(INFO)<<"Summary: mean testing Lab error "<<testing_error_sum / testing_error.size();

	global_ftr_db->Close();
}
