#ifndef CAFFE_COPY_PIPELINE_HPP_
#define CAFFE_COPY_PIPELINE_HPP_

#include "caffe/common.hpp"
#include <map>
#include <vector>

namespace caffe {

template <typename Dtype>
class Blob;

template <typename Dtype>
class IBroadcastDiffNetwork {
public:
	explicit IBroadcastDiffNetwork(std::vector<int> &devices, int src_device):
	src_device_(src_device){}
	virtual ~IBroadcastDiffNetwork(){}

	virtual void BroadcastGpuDiff(std::map<int, shared_ptr<Blob<Dtype> > > &shards,
			Dtype scale_src, Dtype scale_tgt) = 0;

	static IBroadcastDiffNetwork<Dtype>* make(std::vector<int> &devices, int src_device);

protected:
	int src_device_;

  DISABLE_COPY_AND_ASSIGN(IBroadcastDiffNetwork);

};


template <typename Dtype>
class NullDiffBroadcaster: public IBroadcastDiffNetwork<Dtype>{
public:
	explicit NullDiffBroadcaster(std::vector<int> &devices, int src_device):
	IBroadcastDiffNetwork<Dtype>(devices, src_device){

	}

	void BroadcastGpuDiff(std::map<int, shared_ptr<Blob<Dtype> > > &shards,
				Dtype scale_src, Dtype scale_tgt){}

protected:

  DISABLE_COPY_AND_ASSIGN(NullDiffBroadcaster);
};

template <typename Dtype>
class TwoPeeringGPUsDiffBroadcaster: public IBroadcastDiffNetwork<Dtype>{
public:
	explicit TwoPeeringGPUsDiffBroadcaster(std::vector<int> &devices, int src_device);
	~TwoPeeringGPUsDiffBroadcaster();

	void BroadcastGpuDiff(std::map<int, shared_ptr<Blob<Dtype> > > &shards,
				Dtype scale_src, Dtype scale_tgt);

protected:
	int tgt_device_;
	cudaStream_t tgt_stream_;

  DISABLE_COPY_AND_ASSIGN(TwoPeeringGPUsDiffBroadcaster);
};

template <typename Dtype>
class NaiveDiffBroadcaster: public IBroadcastDiffNetwork<Dtype>{
public:
	explicit NaiveDiffBroadcaster(std::vector<int> &devices, int src_device);
	~NaiveDiffBroadcaster();

	void BroadcastGpuDiff(std::map<int, shared_ptr<Blob<Dtype> > > &shards,
				Dtype scale_src, Dtype scale_tgt);

protected:

  DISABLE_COPY_AND_ASSIGN(NaiveDiffBroadcaster);
};



} // namespace caffe

#endif // CAFFE_COPY_PIPELINE_HPP_
