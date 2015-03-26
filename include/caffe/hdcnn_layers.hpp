// Copyright 2015 Zhicheng Yan

#ifndef CAFFE_HDCNN_LAYERS_HPP_
#define CAFFE_HDCNN_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/neuron_layers.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/* Fine2CoarseProbLayer
 * Take fine classification probabilities as input
 * Aggregate them into coarse classification probabilities
*/
template<typename Dtype>
class Fine2CoarseProbLayer: public Layer<Dtype> {
public:
	explicit Fine2CoarseProbLayer(const LayerParameter& param,
			int replica_id, Net<Dtype> *net) :
			Layer<Dtype>(param, replica_id, net) {
	}
//	virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
//			vector<Blob<Dtype>*>* top);
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
				const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
				const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "Fine2CoarseProb"; }

	virtual inline int ExactNumBottomBlobs() const {
		return 1;
	}
	virtual inline int ExactNumTopBlobs() const {
		return 1;
	}

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
				const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
				const vector<bool>& propagate_down,
				const vector<Blob<Dtype>*>& bottom);

	int num_fine_;
	int num_coarse_;
	std::vector<int> fine2coarse_;
	std::vector<vector<int> > coarse2fine_;
};


/* MultinomialLogisticSparsityLossLayer
*/
template <typename Dtype>
class MultinomialLogisticSparsityLossLayer : public Layer<Dtype> {
 public:
  explicit MultinomialLogisticSparsityLossLayer(const LayerParameter& param,
  		int replica_id, Net<Dtype> *net)
      : Layer<Dtype>(param, replica_id, net) {}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
				const vector<Blob<Dtype>*>& top);
  virtual void Reshape(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MultinomialLogisticSparsityLoss"; }

  virtual inline int ExactNumBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int num_branch_;
  shared_ptr<Blob<Dtype> > branch_sparsity_diff_;
};

template<typename Dtype>
class CompactProbabilisticAverageProbLayer: public Layer<Dtype> {
public:
	explicit CompactProbabilisticAverageProbLayer(const LayerParameter& param,
			int replica_id, Net<Dtype> *net):
	Layer<Dtype>(param, replica_id, net){}
	virtual ~CompactProbabilisticAverageProbLayer(){}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
				const vector<Blob<Dtype>*>& top);
  virtual void Reshape(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	int prob_num_;
	int class_num_;
	std::vector<int> compact_layer_size_;
	std::vector<std::vector<int> > compact_layer_class_id_;
};


}  // namespace caffe

#endif  // CAFFE_HDCNN_LAYERS_HPP_
