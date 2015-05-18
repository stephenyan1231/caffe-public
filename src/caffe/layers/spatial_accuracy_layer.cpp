// Copyright 2015 Zhicheng Yan

#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/spatial_accuracy_layer.hpp"

namespace caffe {

template<typename Dtype>
void SpatialAccuracyLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_.reset(LayerRegistry<Dtype>::CreateLayer(softmax_param, this->replica_id_, this->net_));
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);


	top_k_ = this->layer_param_.spatial_accuracy_param().top_k();
	spatial_subsample_height_factor_ = this->layer_param_.spatial_accuracy_param().spatial_subsample_height_factor();
	spatial_subsample_width_factor_ = this->layer_param_.spatial_accuracy_param().spatial_subsample_width_factor();

}

template<typename Dtype>
void SpatialAccuracyLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);

	CHECK_EQ(bottom[0]->num(), bottom[1]->num())
			<< "The data and label should have the same number.";
	CHECK_LE(top_k_, bottom[0]->channels())
			<< "top_k must be less than or equal to the number of classes.";
	CHECK_EQ(bottom[1]->channels(), 1);
	CHECK_EQ(bottom[1]->height(), 1);
	CHECK_EQ(bottom[1]->width(), 1);
	if(bottom.size() > 2){
		CHECK_EQ(bottom[2]->num(), bottom[1]->num());
		CHECK_EQ(bottom[2]->channels(), 1);
		CHECK_EQ(bottom[2]->height(), 1);
		CHECK_EQ(bottom[2]->width(), 2);
	}
	top[0]->Reshape(1, 1, 1, 1);
}

template<typename Dtype>
void SpatialAccuracyLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);

	Dtype accuracy = 0;
	const Dtype* prob_data = prob_.cpu_data();
	DLOG(INFO)<<"SpatialAccuracyLayer<Dtype>::Forward_cpu prob_data shape "<<
			prob_.num()<<" "<<prob_.channels()<<" "<<prob_.height()<<" "<<prob_.width();
	const Dtype* bottom_label = bottom[1]->cpu_data();
	int num = bottom[0]->num();
	int dim = bottom[0]->channels();
	int h = bottom[0]->height();
	int w = bottom[0]->width();
	const Dtype* image_size_data = NULL;
	if(bottom.size() > 2){
		image_size_data = bottom[2]->cpu_data();

	}
	int eh = h, ew = w;

	vector < Dtype > maxval(top_k_ + 1);
	vector<int> max_id(top_k_ + 1);
	for (int i = 0; i < num; ++i) {
		if(bottom.size() > 2){
			eh = (image_size_data+bottom[2]->offset(i))[0] / spatial_subsample_height_factor_;
			ew = (image_size_data+bottom[2]->offset(i))[1] / spatial_subsample_width_factor_;
		}

		std::vector < std::pair<Dtype, int> > bottom_data_vector(dim);
		for(int j = 0;j<dim;++j){
			bottom_data_vector[j].first = 0;
			bottom_data_vector[j].second = j;
		}

		for (int y = 0; y < eh; ++y) {
			for (int x = 0; x < ew; ++x) {
				// Top-k accuracy
				for (int j = 0; j < dim; ++j) {
					bottom_data_vector[j].first += prob_data[(j * h + y) * w + x];
				}
			}
		}
		// can skip the step of dividing the accumulated probabilities by (eh * ew)
		std::partial_sort(bottom_data_vector.begin(),
				bottom_data_vector.begin() + top_k_, bottom_data_vector.end(),
				std::greater<std::pair<Dtype, int> >());
		// check if true label is in top k predictions
		for (int k = 0; k < top_k_; k++) {
			if (bottom_data_vector[k].second == static_cast<int>(bottom_label[i])) {
				++accuracy;
				break;
			}
		}

		prob_data += prob_.offset(1);
	}
	top[0]->mutable_cpu_data()[0] = accuracy / (num);
	// Accuracy layer should not be used as a loss function.

	if (Caffe::phase() == Caffe::TEST && this->conserve_gpu_memory_test_) {
		prob_.ReshapeForceMemoryFree(0, 0, 0, 0);
		bottom[0]->ReshapeForceMemoryFree(0, 0, 0, 0);
	}

}

INSTANTIATE_CLASS(SpatialAccuracyLayer);
REGISTER_LAYER_CLASS(SpatialAccuracy);
} // namespace caffe
