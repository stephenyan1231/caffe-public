// Copyright 2014 Zhicheng Yan

#ifndef CAFFE_ENSEMBLE_PROB_LAYERS_HPP_
#define CAFFE_ENSEMBLE_PROB_LAYERS_HPP_

#include "caffe/layer.hpp"
#include <vector>

namespace caffe {

template<typename Dtype>
class ProbabilisticAverageProbLayer: public Layer<Dtype> {
public:
	explicit ProbabilisticAverageProbLayer(const LayerParameter& param) :
			Layer<Dtype>(param) {
	}
	virtual ~ProbabilisticAverageProbLayer();
	virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);

protected:
	virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);
//	virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
//			vector<Blob<Dtype>*>* top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down, vector<Blob<Dtype>*>* bottom);
//	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
//			const bool propagate_down, vector<Blob<Dtype>*>* bottom);

	int prob_num_;
	int class_num_;
	Dtype** bottom_prob_ptrs_;
	Dtype** bottom_prob_ptrs_dev_;
	Dtype** bottom_prob_diff_ptrs_;
	Dtype** bottom_prob_diff_ptrs_dev_;
};

} // namespace caffe

#endif
