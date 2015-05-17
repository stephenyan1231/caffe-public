#ifndef VIDEO_ENHANCEMENT_UTIL_HPP_
#define VIDEO_ENHANCEMENT_UTIL_HPP_

#include <vector>
#include <map>
#include <string>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"

using namespace std;

typedef vector<int> vecInt;

void read_image_names_from_list(const char* image_list_file,
		std::vector<std::string>& img_names);

caffe::Datum read_LAB_image_from_database(std::string database_path,
		std::string image_name);

void read_in_LAB_images_from_database(std::string database_path,
		const std::vector<std::string> &img_names,
		std::map<std::string, caffe::Datum>& themap);

void get_segment_median_center(vecInt* segment, int img_width, int img_height,
		int &seg_x, int &seg_y);

#endif
