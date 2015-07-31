#include <opencv2/core/core.hpp>

#include <stdint.h>

#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/data_layers_more.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<typename Dtype>
SemanticLabelingDataLayer<Dtype>::SemanticLabelingDataLayer(const LayerParameter& param) :
		BasePrefetchingDataLayer<Dtype>(param), semantic_labeling_transform_param_(param.semantic_labeling_transform_param()) {

}

template<typename Dtype>
SemanticLabelingDataLayer<Dtype>::~SemanticLabelingDataLayer() {
	this->JoinPrefetchThread();
}

template<typename Dtype>
void SemanticLabelingDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	semantic_labeling_data_transformer_.reset(
			new SemanticLabelingDataTransformer<Dtype>(semantic_labeling_transform_param_, this->phase_));
	semantic_labeling_data_transformer_->InitRand();
	// Initialize DB
	db_.reset(
			db::GetDB(this->layer_param_.semantic_labeling_data_param().backend()));
	db_->Open(this->layer_param_.semantic_labeling_data_param().source(),
			db::READ);
	cursor_.reset(db_->NewCursor());

	// Check if we should randomly skip a few data points
	if (this->layer_param_.semantic_labeling_data_param().rand_skip()) {
		unsigned int skip = caffe_rng_rand()
		% this->layer_param_.semantic_labeling_data_param().rand_skip();
		LOG(INFO)<< "Skipping first " << skip << " data points.";
		while (skip-- > 0) {
			cursor_->Next();
		}
	}
	// Read a data point, and use it to initialize the top blob.
	SemanticLabelingDatum datum;
	datum.ParseFromString(cursor_->value());

	int crop_height =
	this->layer_param_.semantic_labeling_transform_param().crop_height();
	int crop_width =
	this->layer_param_.semantic_labeling_transform_param().crop_width();
	int batch_size =
	this->layer_param_.semantic_labeling_data_param().batch_size();
	if (crop_height > 0 || crop_width > 0) {
		CHECK_GT(crop_height, 0);
		CHECK_GT(crop_width, 0);
		top[0]->Reshape(batch_size, datum.channels(), crop_height, crop_width);
		this->prefetch_data_.Reshape(batch_size, datum.channels(), crop_height,
				crop_width);
		this->transformed_data_.Reshape(1, datum.channels(), crop_height,
				crop_width);
		if (this->output_labels_) {
			top[1]->Reshape(batch_size, 1, crop_height, crop_width);
			this->prefetch_label_.Reshape(batch_size, 1, crop_height, crop_width);
			this->transformed_label_.Reshape(1, 1, crop_height, crop_width);
		}
	} else {
		top[0]->Reshape(batch_size, datum.channels(), datum.height(),
				datum.width());
		this->prefetch_data_.Reshape(batch_size, datum.channels(), datum.height(),
				datum.width());
		this->transformed_data_.Reshape(1, datum.channels(), datum.height(),
				datum.width());
		if (this->output_labels_) {
			top[1]->Reshape(batch_size, 1, datum.height(), datum.width());
			this->prefetch_label_.Reshape(batch_size, 1, datum.height(),
					datum.width());
			this->transformed_label_.Reshape(1, 1, datum.height(), datum.width());
		}
	}

}

// This function is used to create a thread that prefetches the data for scene labeling.
template<typename Dtype>
void SemanticLabelingDataLayer<Dtype>::InternalThreadEntry() {
	CPUTimer batch_timer;
	batch_timer.Start();
	double read_time = 0;
	double trans_time = 0;
	CPUTimer timer;
	CHECK(this->prefetch_data_.count());
	CHECK(this->transformed_data_.count());

	Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
	Dtype* top_label = NULL;  // suppress warnings about uninitialized variables

	if (this->output_labels_) {
		top_label = this->prefetch_label_.mutable_cpu_data();
	}

	const int batch_size = this->layer_param_.semantic_labeling_data_param().batch_size();
	for (int item_id = 0; item_id < batch_size; ++item_id) {
		timer.Start();
		// get a blob
		SemanticLabelingDatum datum;
		datum.ParseFromString(cursor_->value());

		cv::Mat cv_img;
		if (datum.encoded()) {
			cv_img = DecodeSemanticLabelingDatumToCVMatNative(datum);
			if (cv_img.channels() != this->transformed_data_.channels()) {
				LOG(WARNING)<< "Your dataset contains encoded images with mixed "
				<< "channel sizes. Consider adding a 'force_color' flag to the "
				<< "model definition, or rebuild your dataset using "
				<< "convert_imageset.";
			}
		}
		read_time += timer.MicroSeconds();
		timer.Start();

		// Apply data transformations (mirror, scale, crop...)
		int offset_data = this->prefetch_data_.offset(item_id);
		int offset_label = this->prefetch_label_.offset(item_id);
		this->transformed_data_.set_cpu_data(top_data + offset_data);
		if (this->output_labels_) {
			this->transformed_label_.set_cpu_data(top_label + offset_label);
		}
		if (datum.encoded()) {
			if (this->output_labels_) {
				this->semantic_labeling_data_transformer_->Transform(datum, cv_img, &(this->transformed_data_), &(this->transformed_label_));
			} else {
				this->semantic_labeling_data_transformer_->Transform(datum, cv_img, &(this->transformed_data_), NULL);
			}

		} else {
			if (this->output_labels_) {
				this->semantic_labeling_data_transformer_->Transform(datum, &(this->transformed_data_), &(this->transformed_label_));
			} else {
				this->semantic_labeling_data_transformer_->Transform(datum, &(this->transformed_data_), NULL);
			}
		}
		trans_time += timer.MicroSeconds();
		// go to the next iter
		cursor_->Next();
		if (!cursor_->valid()) {
			DLOG(INFO)<< "Restarting data prefetching from start.";
			cursor_->SeekToFirst();
		}
	} // for (int item_id = 0; item_id < batch_size; ++item_id)
	batch_timer.Stop();
	DLOG(INFO)<< "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
	DLOG(INFO)<< "     Read time: " << read_time / 1000 << " ms.";
	DLOG(INFO)<< "Transform time: " << trans_time / 1000 << " ms.";
}

//template<typename Dtype>
//void SemanticLabelingDataLayer<Dtype>::Fill_label_weight_map(const Blob<Dtype> &label_map, Blob<Dtype> &label_weight_map) {
//	CHECK_EQ(label_map.shape(0), label_weight_map.shape(0));
//	CHECK_EQ(label_map.shape(2), label_weight_map.shape(2));
//	CHECK_EQ(label_map.shape(3), label_weight_map.shape(3));
//	const Dtype *label_data = label_map.cpu_data();
//	Dtype* label_weight_data = label_weight_map.mutable_cpu_data();
//	for (int i = 0; i < label_map.count(); ++i) {
//		int label = static_cast<int>(label_data[i]);
//		CHECK(label < label_weights_.size());
//		label_weight_data[i] = label_weights_[label];
//	}
//}

template<typename Dtype>
void SemanticLabelingDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	// First, join the thread
	this->JoinPrefetchThread();
	DLOG(INFO)<< "Thread joined";
	// Reshape to loaded data.
	top[0]->Reshape(this->prefetch_data_.num(), this->prefetch_data_.channels(), this->prefetch_data_.height(), this->prefetch_data_.width());
	// Copy the data
	caffe_copy(this->prefetch_data_.count(), this->prefetch_data_.cpu_data(), top[0]->mutable_cpu_data());
	DLOG(INFO)<< "Prefetch copied";
	if (this->output_labels_) {
		caffe_copy(this->prefetch_label_.count(), this->prefetch_label_.cpu_data(), top[1]->mutable_cpu_data());
	}
	// Start a new prefetch thread
	DLOG(INFO)<< "CreatePrefetchThread";
	this->CreatePrefetchThread();
}

INSTANTIATE_CLASS(SemanticLabelingDataLayer);
REGISTER_LAYER_CLASS(SemanticLabelingData);

} // namespace caffe
