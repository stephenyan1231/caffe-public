// Copyright 2015 Zhicheng Yan

#include <vector>
#include <algorithm>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/hdcnn_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
void Fine2MultiCoarseProbLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	num_coarse_ =
			this->layer_param_.fine2multicoarse_prob_layer_param().num_coarse();
//	num_fine_ = bottom[0]->count() / bottom[0]->num();
	num_fine_ =
			this->layer_param_.fine2multicoarse_prob_layer_param().fine2multicoarse_param_size();
	fine2multicoarse_.resize(num_fine_);
	coarse2fine_.resize(num_coarse_);
	coarse2fine_not_.resize(num_coarse_);

	for (int i = 0; i < num_fine_; ++i) {
		Fine2MultiCoarseParameter fine2multicoarse_param =
				this->layer_param_.fine2multicoarse_prob_layer_param().fine2multicoarse_param(
						i);
		for (int j = 0; j < fine2multicoarse_param.coarse_id_size(); ++j) {
			fine2multicoarse_[i].push_back(fine2multicoarse_param.coarse_id(j));
			coarse2fine_[fine2multicoarse_param.coarse_id(j)].push_back(i);
		}
//		coarse2fine_[fine2coarse_[i]].push_back(i);
	}

	for (int i = 0; i < num_coarse_; ++i) {
		std::vector<int> full_fine;
		for (int j = 0; j < num_fine_; ++j) {
			full_fine.push_back(j);
		}
		coarse2fine_not_[i].resize(num_fine_);
		std::sort(coarse2fine_[i].begin(),coarse2fine_[i].end());
		std::sort(full_fine.begin(),full_fine.end());

		std::vector<int>::iterator it = std::set_difference(full_fine.begin(),
				full_fine.end(), coarse2fine_[i].begin(), coarse2fine_[i].end(),
				coarse2fine_not_[i].begin());
		coarse2fine_not_[i].resize(it - coarse2fine_not_[i].begin());
		LOG(ERROR) << "coarse2fine_not_ " << i << " size "
				<< coarse2fine_not_[i].size();
	}

}

template<typename Dtype>
void Fine2MultiCoarseProbLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	if (bottom[0]->count() > 0) {
		CHECK_EQ(bottom[0]->channels(), num_fine_);
//		CHECK_EQ(bottom[0]->count() / bottom[0]->num(), num_fine_);
	}
	top[0]->Reshape(bottom[0]->num(), num_coarse_, bottom[0]->height(),
			bottom[0]->width());
	unnormalized_coarse_prob_.Reshape(bottom[0]->num(), num_coarse_,
			bottom[0]->height(), bottom[0]->width());
	coarse_prob_sum_.Reshape(bottom[0]->num(), 1, bottom[0]->height(),
			bottom[0]->width());
}

template<typename Dtype>
void Fine2MultiCoarseProbLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	memset(top_data, 0, sizeof(Dtype) * top[0]->count());

	Dtype *unnormalized_coarse_prob_data_ =
			unnormalized_coarse_prob_.mutable_cpu_data();
	memset(unnormalized_coarse_prob_data_, 0,
			sizeof(Dtype) * unnormalized_coarse_prob_.count());

	for (int i = 0; i < bottom[0]->num(); ++i) {
		for (int j = 0; j < num_fine_; ++j) {
			for (int k = 0; k < fine2multicoarse_[j].size(); ++k) {
				for (int y = 0; y < bottom[0]->height(); ++y) {
					for (int x = 0; x < bottom[0]->width(); ++x) {
						(unnormalized_coarse_prob_data_)[unnormalized_coarse_prob_.offset(i,
								fine2multicoarse_[j][k], y, x)] +=
								bottom_data[bottom[0]->offset(i, j, y, x)];
						//					(top_data)[top[0]->offset(i) + fine2coarse_[j]] +=
						//							bottom_data[bottom[0]->offset(i) + j];
					}
				}
			}
		}
	}

	Dtype* coarse_prob_sum_data = coarse_prob_sum_.mutable_cpu_data();
	caffe_memset(sizeof(Dtype) * coarse_prob_sum_.count(), 0,
			coarse_prob_sum_data);
	for (int i = 0; i < bottom[0]->num(); ++i) {
		for (int j = 0; j < num_coarse_; ++j) {
			for (int y = 0; y < bottom[0]->height(); ++y) {
				for (int x = 0; x < bottom[0]->width(); ++x) {
					coarse_prob_sum_data[coarse_prob_sum_.offset(i, 0, y, x)] +=
							unnormalized_coarse_prob_data_[unnormalized_coarse_prob_.offset(i,
									j, y, x)];
				}
			}
		}
	}

	for (int i = 0; i < bottom[0]->num(); ++i) {
		for (int j = 0; j < num_coarse_; ++j) {
			for (int y = 0; y < bottom[0]->height(); ++y) {
				for (int x = 0; x < bottom[0]->width(); ++x) {
					top_data[top[0]->offset(i, j, y, x)] =
							unnormalized_coarse_prob_data_[unnormalized_coarse_prob_.offset(i,
									j, y, x)]
									/ coarse_prob_sum_data[coarse_prob_sum_.offset(i, 0, y, x)];
				}
			}
		}
	}
}

template<typename Dtype>
void Fine2MultiCoarseProbLayer<Dtype>::Backward_cpu(
		const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
	const Dtype *top_diff = top[0]->cpu_diff();
	const Dtype *unnormalized_coarse_prob_data =
			unnormalized_coarse_prob_.cpu_data();
	const Dtype* coarse_prob_sum_data = coarse_prob_sum_.cpu_data();
	if (propagate_down[0]) {
		Dtype *bottom_diff = bottom[0]->mutable_cpu_diff();
		caffe_memset(sizeof(Dtype) * bottom[0]->count(), 0, bottom_diff);

		for (int i = 0; i < bottom[0]->num(); ++i) {
			for (int j = 0; j < num_coarse_; ++j) {
				for (int y = 0; y < bottom[0]->height(); ++y) {
					for (int x = 0; x < bottom[0]->width(); ++x) {
						Dtype coarse_prob_sum =
								coarse_prob_sum_data[coarse_prob_sum_.offset(i, 0, y, x)];

						for (int k = 0; k < coarse2fine_[j].size(); ++k) {
							bottom_diff[bottom[0]->offset(i, coarse2fine_[j][k], y, x)] +=
									((coarse_prob_sum
											- unnormalized_coarse_prob_data[unnormalized_coarse_prob_.offset(
													i, j, y, x)]
													* fine2multicoarse_[coarse2fine_[j][k]].size())
											/ (coarse_prob_sum * coarse_prob_sum))*top_diff[top[0]->offset(i,j,y,x)];
						}

						for (int k = 0; k < coarse2fine_not_[j].size(); ++k) {
							bottom_diff[bottom[0]->offset(i, coarse2fine_not_[j][k], y, x)] +=
									((-unnormalized_coarse_prob_data[unnormalized_coarse_prob_.offset(
											i, j, y, x)]
											* fine2multicoarse_[coarse2fine_not_[j][k]].size())
											/ (coarse_prob_sum * coarse_prob_sum))*top_diff[top[0]->offset(i,j,y,x)];
						}

					}
				}
//				for (int k = 0; k < coarse2fine_[j].size(); ++k) {
//					bottom_diff[bottom[0]->offset(i) + coarse2fine_[j][k]] =
//							top_diff[top[0]->offset(i) + j];
//				}
			}
		}
	}
}

INSTANTIATE_CLASS(Fine2MultiCoarseProbLayer);
REGISTER_LAYER_CLASS(Fine2MultiCoarseProb);

}  // namespace caffe
