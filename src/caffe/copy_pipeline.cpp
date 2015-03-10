#include "caffe/copy_pipeline.hpp"
#include "caffe/blob.hpp"

namespace caffe {

template<typename Dtype>
IBroadcastDiffNetwork<Dtype>* IBroadcastDiffNetwork<Dtype>::make(
		std::vector<int> &devices, int src_device) {
//	LOG(INFO)<<"IBroadcastDiffNetwork<Dtype>* IBroadcastDiffNetwork";
	if (devices.size() == 1) {
		return new NullDiffBroadcaster<Dtype>(devices, src_device);
	} else if (devices.size() == 2) {
		int tgt_device = devices[0] == src_device ? devices[1] : devices[0];
		if (Caffe::CanAccessPeer(tgt_device, src_device)) {
			return new TwoPeeringGPUsDiffBroadcaster<Dtype>(devices, src_device);
		} else {
			return new NaiveDiffBroadcaster<Dtype>(devices, src_device);
		}
	} else {
		return new NaiveDiffBroadcaster<Dtype>(devices, src_device);
	}
}

INSTANTIATE_CLASS(IBroadcastDiffNetwork);

INSTANTIATE_CLASS(NullDiffBroadcaster);

template<typename Dtype>
TwoPeeringGPUsDiffBroadcaster<Dtype>::TwoPeeringGPUsDiffBroadcaster(
		std::vector<int> &devices, int src_device) :
		IBroadcastDiffNetwork<Dtype>(devices, src_device) {
	CHECK_EQ(devices.size(), 2);
	tgt_device_ = devices[0] == src_device ? devices[1] : devices[0];
}

template<typename Dtype>
TwoPeeringGPUsDiffBroadcaster<Dtype>::~TwoPeeringGPUsDiffBroadcaster() {
}

template<typename Dtype>
void TwoPeeringGPUsDiffBroadcaster<Dtype>::BroadcastGpuDiff(
		std::map<int, shared_ptr<Blob<Dtype> > > &shards, Dtype scale_src,
		Dtype scale_tgt) {
	int old_device = Caffe::GetDeviceId();
	Caffe::SetDevice(tgt_device_);
	CHECK_GT(shards[this->src_device_]->count(), 0);
	CHECK_EQ(shards[this->src_device_]->count(), shards[tgt_device_]->count());
	caffe_gpu_axpby<Dtype>(shards[tgt_device_]->count(), scale_src,
			shards[this->src_device_]->gpu_diff(), scale_tgt,
			shards[tgt_device_]->mutable_gpu_diff());
	Caffe::SetDevice(old_device);
}

INSTANTIATE_CLASS(TwoPeeringGPUsDiffBroadcaster);

template<typename Dtype>
NaiveDiffBroadcaster<Dtype>::NaiveDiffBroadcaster(std::vector<int> &devices,
		int src_device) :
		IBroadcastDiffNetwork<Dtype>(devices, src_device) {

}

template<typename Dtype>
NaiveDiffBroadcaster<Dtype>::~NaiveDiffBroadcaster() {

}

template<typename Dtype>
void NaiveDiffBroadcaster<Dtype>::BroadcastGpuDiff(
		std::map<int, shared_ptr<Blob<Dtype> > > &shards, Dtype scale_src,
		Dtype scale_tgt) {
// TO DO

}

INSTANTIATE_CLASS(NaiveDiffBroadcaster);

} // namespace caffe
