#include <opencv2/core/core.hpp>

#include <string>
#include <vector>

#include "caffe/semantic_labeling_data_transformer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<typename Dtype>
SemanticLabelingDataTransformer<Dtype>::SemanticLabelingDataTransformer(
		const SemanticLabelingTransformationParameter& param, Phase phase) :
		param_(param), phase_(phase) {
	// check if we want to use mean_value
	if (param_.mean_value_size() > 0) {
		for (int c = 0; c < param_.mean_value_size(); ++c) {
			mean_values_.push_back(param_.mean_value(c));
		}
	}
}

template<typename Dtype>
void SemanticLabelingDataTransformer<Dtype>::InitRand() {
	const bool needs_rand = param_.mirror()
			|| (phase_ == TRAIN && param_.crop_height() && param_.crop_width());
	if (needs_rand) {
		const unsigned int rng_seed = caffe_rng_rand();
		rng_.reset(new Caffe::RNG(rng_seed));
	} else {
		rng_.reset();
	}
}

template<typename Dtype>
void SemanticLabelingDataTransformer<Dtype>::Transform(
		const SemanticLabelingDatum& datum, Blob<Dtype>* transformed_blob,
		Blob<Dtype>* transformed_label) {
	const int datum_channels = datum.channels();
	const int datum_height = datum.height();
	const int datum_width = datum.width();

	const int channels = transformed_blob->channels();
	const int height = transformed_blob->height();
	const int width = transformed_blob->width();
	const int num = transformed_blob->num();

	if (transformed_label) {
		CHECK_EQ(height, transformed_label->shape(2));
		CHECK_EQ(width, transformed_label->shape(3));
	}

	CHECK_EQ(channels, datum_channels);
	CHECK_LE(height, datum_height);
	CHECK_LE(width, datum_width);
	CHECK_GE(num, 1);

	const int crop_height = param_.crop_height();
	const int crop_width = param_.crop_width();
	if (crop_height || crop_width) {
		CHECK_GT(crop_height, 0);
		CHECK_GT(crop_width, 0);
		CHECK_EQ(crop_height, height);
		CHECK_EQ(crop_width, width);
	} else {
		CHECK_EQ(datum_height, height);
		CHECK_EQ(datum_width, width);
	}

	if (transformed_label) {
		Transform(datum, transformed_blob->mutable_cpu_data(),
				transformed_label->mutable_cpu_data());
	} else {
		Transform(datum, transformed_blob->mutable_cpu_data(), NULL);
	}

}

template<typename Dtype>
void SemanticLabelingDataTransformer<Dtype>::Transform(
		const SemanticLabelingDatum& datum, const cv::Mat& cv_img,
		Blob<Dtype>* transformed_blob, Blob<Dtype>* transformed_label) {
	const int img_channels = cv_img.channels();
	const int img_height = cv_img.rows;
	const int img_width = cv_img.cols;

	const int channels = transformed_blob->channels();
	const int height = transformed_blob->height();
	const int width = transformed_blob->width();
	const int num = transformed_blob->num();

	CHECK_EQ(channels, img_channels);
	CHECK_LE(height, img_height);
	CHECK_LE(width, img_width);
	CHECK_GE(num, 1);

	CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";

	const int crop_height = param_.crop_height();
	const int crop_width = param_.crop_width();
	const Dtype scale = param_.scale();
//  const bool do_mirror = param_.mirror() && Rand(2);
	const bool do_mirror = false;  // disable mirroring for the time being

	const bool has_mean_values = mean_values_.size() > 0;

	CHECK_GT(img_channels, 0);
	CHECK_GE(img_height, crop_height);
	CHECK_GE(img_width, crop_width);
	if (has_mean_values) {
		CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels)
				<< "Specify either 1 mean_value or as many as channels: "
				<< img_channels;
		if (img_channels > 1 && mean_values_.size() == 1) {
			// Replicate the mean_value for simplicity
			for (int c = 1; c < img_channels; ++c) {
				mean_values_.push_back(mean_values_[0]);
			}
		}
	}

	int h_off = 0;
	int w_off = 0;
	cv::Mat cv_cropped_img = cv_img;
	if (crop_height || crop_width) {
		CHECK_GT(crop_height, 0);
		CHECK_GT(crop_width, 0);
		CHECK_EQ(crop_height, height);
		CHECK_EQ(crop_width, width);
		// We only do random crop when we do training.
		if (phase_ == TRAIN) {
			h_off = Rand(img_height - crop_height + 1);
			w_off = Rand(img_width - crop_width + 1);
		} else {
			h_off = (img_height - crop_height) / 2;
			w_off = (img_width - crop_width) / 2;
		}
		cv::Rect roi(w_off, h_off, crop_width, crop_height);
		cv_cropped_img = cv_img(roi);
	} else {
		CHECK_EQ(img_height, height);
		CHECK_EQ(img_width, width);
	}
	CHECK(cv_cropped_img.data);

	Dtype* transformed_data = transformed_blob->mutable_cpu_data();
	Dtype* transformed_label_data = NULL;
	if(transformed_label) {
		transformed_label_data = transformed_label->mutable_cpu_data();
	}
	int top_index;
	for (int h = 0; h < height; ++h) {
		const uchar* ptr = cv_cropped_img.ptr < uchar > (h);
		int img_index = 0;
		for (int w = 0; w < width; ++w) {
			if(transformed_label){
				if (do_mirror) {
					top_index = h * width + (width - 1 - w);
				} else {
					top_index = h * width + w;
				}
				int label_index = (h + h_off) * img_width + w + w_off;
				CHECK_LT(label_index, datum.label_size());
				transformed_label_data[top_index] = datum.label(label_index);
			}

			for (int c = 0; c < img_channels; ++c) {
				if (do_mirror) {
					top_index = (c * height + h) * width + (width - 1 - w);
				} else {
					top_index = (c * height + h) * width + w;
				}
				Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
				if (has_mean_values) {
					transformed_data[top_index] = (pixel - mean_values_[c]) * scale;
				} else {
					transformed_data[top_index] = pixel * scale;
				}
			}
		}
	}
}

template<typename Dtype>
int SemanticLabelingDataTransformer<Dtype>::Rand(int n) {
	CHECK(rng_);
	CHECK_GT(n, 0);
	caffe::rng_t* rng = static_cast<caffe::rng_t*>(rng_->generator());
	return ((*rng)() % n);
}

template<typename Dtype>
void SemanticLabelingDataTransformer<Dtype>::Transform(
		const SemanticLabelingDatum& datum, Dtype* transformed_data,
		Dtype* transformed_label) {
	const string& data = datum.data();
	const int datum_channels = datum.channels();
	const int datum_height = datum.height();
	const int datum_width = datum.width();

	const int crop_height = param_.crop_height();
	const int crop_width = param_.crop_width();
	const Dtype scale = param_.scale();
//  const bool do_mirror = param_.mirror() && Rand(2);
	const bool do_mirror = false;  // disable mirroring for the time being
	const bool has_uint8 = data.size() > 0;
	const bool has_mean_values = mean_values_.size() > 0;

	CHECK_GT(datum_channels, 0);
	CHECK_GE(datum_height, crop_height);
	CHECK_GE(datum_width, crop_width);

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

	int h_off = 0;
	int w_off = 0;
	if (crop_height || crop_width) {
		CHECK_GT(crop_height, 0);
		CHECK_GT(crop_width, 0);
		height = crop_height;
		width = crop_width;
		// We only do random crop when we do training.
		if (phase_ == TRAIN) {
			h_off = Rand(datum_height - crop_height + 1);
			w_off = Rand(datum_width - crop_width + 1);
		} else {
			h_off = (datum_height - crop_height) / 2;
			w_off = (datum_width - crop_width) / 2;
		}
	}

	Dtype datum_element;
	int top_index, data_index;
	for (int c = 0; c < datum_channels; ++c) {
		for (int h = 0; h < height; ++h) {
			for (int w = 0; w < width; ++w) {
				data_index = (c * datum_height + h_off + h) * datum_width + w_off + w;
				if (do_mirror) {
					top_index = (c * height + h) * width + (width - 1 - w);
				} else {
					top_index = (c * height + h) * width + w;
				}
				if(transformed_label){
					transformed_label[top_index] = datum.label(data_index);
				}
				if (has_uint8) {
					datum_element =
							static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
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

INSTANTIATE_CLASS(SemanticLabelingDataTransformer);

} // namespace caffe
