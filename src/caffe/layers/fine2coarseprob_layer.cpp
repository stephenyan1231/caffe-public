// Copyright 2015 Zhicheng Yan

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/hdcnn_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template<typename Dtype>
void Fine2CoarseProbLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {
	num_coarse_ = this->layer_param_.fine2coarse_prob_layer_param().num_coarse();
	num_fine_ = bottom[0]->count() / bottom[0]->num();
	CHECK_EQ(
			this->layer_param_.fine2coarse_prob_layer_param().fine2coarse_size(),
			num_fine_);
	fine2coarse_.resize(num_fine_);
	coarse2fine_.resize(num_coarse_);
	for (int i = 0; i < num_fine_; ++i) {
		fine2coarse_[i] =
				this->layer_param_.fine2coarse_prob_layer_param().fine2coarse(i);
		coarse2fine_[fine2coarse_[i]].push_back(i);
	}
}

template<typename Dtype>
void Fine2CoarseProbLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	top[0]->Reshape(bottom[0]->num(), num_coarse_, 1, 1);
}

template<typename Dtype>
void Fine2CoarseProbLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	memset(top_data, 0, sizeof(Dtype) * top[0]->count());

	for (int i = 0; i < bottom[0]->num(); ++i) {
		for (int j = 0; j < num_fine_; ++j) {
			(top_data)[top[0]->offset(i) + fine2coarse_[j]] +=
					bottom_data[bottom[0]->offset(i) + j];
		}
	}
}

template<typename Dtype>
void Fine2CoarseProbLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
	const Dtype *top_diff = top[0]->cpu_diff();
	if (propagate_down[0]) {
		Dtype *bottom_diff = bottom[0]->mutable_cpu_diff();
		for (int i = 0; i < bottom[0]->num(); ++i) {
			for (int j = 0; j < num_coarse_; ++j) {
				for (int k = 0; k < coarse2fine_[j].size(); ++k) {
					bottom_diff[bottom[0]->offset(i) + coarse2fine_[j][k]] =
							top_diff[top[0]->offset(i) + j];
				}
			}
		}
	}
}

INSTANTIATE_CLASS(Fine2CoarseProbLayer);
REGISTER_LAYER_CLASS(Fine2CoarseProb);

}  // namespace caffe
