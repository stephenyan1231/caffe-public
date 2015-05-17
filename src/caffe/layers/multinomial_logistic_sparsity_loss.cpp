// Copyright 2014 Zhicheng Yan@eBay

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"
#include "caffe/hdcnn_layers.hpp"

using std::max;

namespace caffe {

template<typename Dtype>
void MultinomialLogisticSparsityLossLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
	Layer<Dtype>::LayerSetUp(bottom, top);
	CHECK_EQ(bottom[0]->num(), bottom[1]->num())
	  << "The data and label should have the same number.";

	CHECK_EQ(bottom[1]->channels(), 1);
	CHECK_EQ(bottom[1]->height(), 1);
	CHECK_EQ(bottom[1]->width(), 1);
	num_branch_ = bottom[2]->count() / bottom[2]->num();
	CHECK_EQ(num_branch_, this->layer_param_.multinomial_logistic_sparsity_loss_param().target_sparsity_size());

	LOG(INFO) << "MultinomialLogisticSparsityLossLayer Forward_cpu "
			<< num_branch_ << " branches";
	branch_sparsity_diff_.reset(new Blob<Dtype>(1, 1, 1, num_branch_));

  // LossLayers have a non-zero (1) loss by default.
  if (this->layer_param_.loss_weight_size() == 0) {
    this->layer_param_.add_loss_weight(Dtype(1));
  }
}

template<typename Dtype>
void MultinomialLogisticSparsityLossLayer<Dtype>::Reshape(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
  top[0]->Reshape(1, 1, 1, 1);
}

template<typename Dtype>
void MultinomialLogisticSparsityLossLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* bottom_label = bottom[1]->cpu_data();
	const Dtype* bottom_branch_prob = bottom[2]->cpu_data();
	Dtype* branch_sparsity_diff_data = branch_sparsity_diff_->mutable_cpu_data();
	int num = bottom[0]->num();
	int dim = bottom[0]->count() / bottom[0]->num();

	Dtype loss = 0;
	memset(branch_sparsity_diff_data, 0,
			sizeof(Dtype) * branch_sparsity_diff_->count());
	for (int i = 0; i < num; ++i) {
		int label = static_cast<int>(bottom_label[i]);
		Dtype prob = max(bottom_data[i * dim + label], Dtype(kLOG_THRESHOLD));
		loss += -log(prob);
//		Dtype coarse_prob_sum = 0;
		for (int j = 0; j < num_branch_; ++j) {
//			coarse_prob_sum +=(bottom_branch_prob + bottom[2]->offset(i))[j];
			branch_sparsity_diff_data[j] +=
					(bottom_branch_prob + bottom[2]->offset(i))[j];
		}
//		LOG(INFO)<<"MultinomialLogisticSparsityLossLayer<Dtype>::Forward_cpu coarse prob sum "<<coarse_prob_sum;
	}
	std::ostringstream branch_sparsity_diff_msg;
	for (int j = 0; j < num_branch_; ++j) {
		branch_sparsity_diff_data[j] = this->layer_param_.multinomial_logistic_sparsity_loss_param().target_sparsity(j)
				- (branch_sparsity_diff_data[j] / num);
		branch_sparsity_diff_msg << branch_sparsity_diff_data[j] <<" ";
	}

	Dtype loss_2nd = 0;
	for (int j = 0; j < num_branch_; ++j) {
		loss_2nd += 0.5 * branch_sparsity_diff_data[j]
				* branch_sparsity_diff_data[j];
	}
	loss_2nd *=
			this->layer_param_.multinomial_logistic_sparsity_loss_param().sparsity_lamda();
	DLOG(INFO)<<"branch_sparsity_diff_data "<<branch_sparsity_diff_msg.str();
	LOG(INFO)<< "\tloss 1st:" << (loss / num) << " loss 2nd: " << (loss_2nd);
	top[0]->mutable_cpu_data()[0] = (loss / num) + (loss_2nd);
}

template<typename Dtype>
void MultinomialLogisticSparsityLossLayer<Dtype>::Backward_cpu(
		const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* bottom_label = bottom[1]->cpu_data();
	const Dtype* branch_sparsity_diff_data = branch_sparsity_diff_->cpu_data();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	Dtype* bottom_branch_prob_diff = bottom[2]->mutable_cpu_diff();
	int num = bottom[0]->num();
	int dim = bottom[0]->count() / bottom[0]->num();
	memset(bottom_diff, 0, sizeof(Dtype) * bottom[0]->count());
	memset(bottom_branch_prob_diff, 0, sizeof(Dtype) * bottom[2]->count());
	for (int i = 0; i < num; ++i) {
		int label = static_cast<int>(bottom_label[i]);
		Dtype prob = max(bottom_data[i * dim + label], Dtype(kLOG_THRESHOLD));
		bottom_diff[i * dim + label] = -1. / (prob * num);

		for (int j = 0; j < num_branch_; ++j) {
			(bottom_branch_prob_diff + bottom[2]->offset(i))[j] =
					branch_sparsity_diff_data[j] * ((Dtype) -1.0 / (num))
							* this->layer_param_.multinomial_logistic_sparsity_loss_param().sparsity_lamda();
		}
	}
}

INSTANTIATE_CLASS(MultinomialLogisticSparsityLossLayer);
REGISTER_LAYER_CLASS(MultinomialLogisticSparsityLoss);

}  // namespace caffe
