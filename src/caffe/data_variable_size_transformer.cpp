#include <opencv2/core/core.hpp>

#include <string>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/data_variable_size_transformer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<typename Dtype>
DataVariableSizeTransformer<Dtype>::DataVariableSizeTransformer(
		const TransformationParameter& param, Phase phase) :
		param_(param), phase_(phase) {
	// check if we want to use mean_file
	if (param_.has_mean_file()) {
		CHECK_EQ(param_.mean_value_size(), 0)<<
		"Cannot specify mean_file and mean_value at the same time";
		const string& mean_file = param.mean_file();
		LOG(INFO) << "Loading mean file from: " << mean_file;
		BlobProto blob_proto;
		ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
		data_mean_.FromProto(blob_proto);
	}
	// check if we want to use mean_value
	if (param_.mean_value_size() > 0) {
		CHECK(param_.has_mean_file() == false) <<
		"Cannot specify mean_file and mean_value at the same time";
		for (int c = 0; c < param_.mean_value_size(); ++c) {
			mean_values_.push_back(param_.mean_value(c));
		}
	}
}

template<typename Dtype>
void DataVariableSizeTransformer<Dtype>::Transform(const Datum& datum,
		Dtype* transformed_data, int max_pixel_num, int& datum_height,
		int& datum_width) {
	int datum_channels = datum.channels();
	datum_height = datum.height();
	datum_width = datum.width();
	int datum_label = datum.label();
	const string *data = &datum.data();

	const int resize_short_side_min = param_.resize_short_side_min();
	const int resize_short_side_max = param_.resize_short_side_max();
	Datum resized_datum;
	if (resize_short_side_min > 0 && resize_short_side_max > 0) {
		// TO DO
		// Now supports only byte data.
		// need to support float data also
		CHECK_GE(resize_short_side_max, resize_short_side_min);
		const int resize_short_side = resize_short_side_min
				+ Rand(resize_short_side_max - resize_short_side_min + 1);
		cv::Mat* cv_origin_img = DatumToCVMat(datum);
		int resize_width = 0, resize_height = 0;
		if (cv_origin_img->rows > cv_origin_img->cols) {
			resize_width = resize_short_side;
			resize_height = ceil(
					(float(cv_origin_img->rows) / float(cv_origin_img->cols))
							* resize_width);
		} else {
			resize_height = resize_short_side;
			resize_width = ceil(
					(float(cv_origin_img->cols) / float(cv_origin_img->rows))
							* resize_height);
		}
		cv::Mat cv_img;
		cv::resize(*cv_origin_img, cv_img, cv::Size(resize_width, resize_height));
		CVMatToDatum(cv_img, &resized_datum);
		resized_datum.set_label(datum_label);

		datum_height = resized_datum.height();
		datum_width = resized_datum.width();
		data = &resized_datum.data();
		delete cv_origin_img;
	}

	CHECK_GE(max_pixel_num, datum_height * datum_width);

//	CHECK_GE(max_height, datum_height);
//	CHECK_GE(max_width, datum_width);

	const Dtype scale = param_.scale();
	bool do_mirror = true;
	if (phase_ == TRAIN) {
		do_mirror = param_.mirror() && Rand(2);
	} else {
		do_mirror = param_.force_mirror();
	}
//	const bool do_mirror = param_.mirror() && Rand(2);
	const bool has_uint8 = (*data).size() > 0;
	const bool has_mean_values = mean_values_.size() > 0;
	CHECK_EQ(has_mean_values, true);

	CHECK_GT(datum_channels, 0);

	Dtype* mean = NULL;

	if (has_mean_values) {
		CHECK(mean_values_.size() == 1 || mean_values_.size() == datum_channels)
																																								<< "Specify either 1 mean_value or as many as channels: "
																																								<< datum_channels;
		if (datum_channels > 1 && mean_values_.size() == 1) {
			// Replicate the mean_value for simplicity
			for (int c = 1; c < datum_channels; ++c) {
				mean_values_.push_back(mean_values_[0]);
			}
		}
	}

	int height = datum_height;
	int width = datum_width;

	// Old version: copy datum image data into the upper-left corner of transformed_data
	// New version: put datum image data in a continuous memory section of transformed_data
	Dtype datum_element;
	int top_index, data_index;
	caffe_memset(sizeof(Dtype) * max_pixel_num * datum_channels, 0,
			transformed_data);
	for (int data_index = 0, c = 0; c < datum_channels; ++c) {
		for (int h = 0; h < height; ++h) {
			for (int w = 0; w < width; ++w, ++data_index) {
				if (do_mirror) {
					top_index = (c * height + h) * width + (width - 1 - w);
				} else {
					top_index = (c * height + h) * width + w;
				}
				if (has_uint8) {
					datum_element =
							static_cast<Dtype>(static_cast<uint8_t>((*data)[data_index]));
				} else {
					datum_element = datum.float_data(data_index);
				}
				if (has_mean_values) {
					transformed_data[top_index] = (datum_element - mean_values_[c])
							* scale;
				} else {
					transformed_data[top_index] = datum_element * scale;
				}
			}
		}
	}
}

template<typename Dtype>
void DataVariableSizeTransformer<Dtype>::Transform(const Datum& datum,
		Blob<Dtype>* transformed_blob, int& datum_height, int& datum_width) {
	const int datum_channels = datum.channels();

	const int channels = transformed_blob->channels();
	const int num = transformed_blob->num();

	CHECK_EQ(channels, datum_channels);
	CHECK_GE(num, 1);

	Dtype* transformed_data = transformed_blob->mutable_cpu_data();
	Transform(datum, transformed_data,
			transformed_blob->height() * transformed_blob->width(), datum_height,
			datum_width);
}

template<typename Dtype>
void DataVariableSizeTransformer<Dtype>::InitRand() {
	const bool needs_rand = param_.mirror()
			|| (phase_ == TRAIN && param_.crop_size())
			|| (param_.resize_short_side_min() > 0
					&& param_.resize_short_side_max() > 0);

	if (needs_rand) {
		const unsigned int rng_seed = caffe_rng_rand();
		rng_.reset(new Caffe::RNG(rng_seed));
	} else {
		rng_.reset();
	}
}

template<typename Dtype>
int DataVariableSizeTransformer<Dtype>::Rand(int n) {
	CHECK(rng_);
	CHECK_GT(n, 0);
	caffe::rng_t* rng = static_cast<caffe::rng_t*>(rng_->generator());
	return ((*rng)() % n);
}

INSTANTIATE_CLASS(DataVariableSizeTransformer);

}  // namespace caffe
