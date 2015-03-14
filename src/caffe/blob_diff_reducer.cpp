#include "caffe/blob_diff_reducer.hpp"
#include "caffe/net.hpp"

namespace caffe {


template<typename Dtype>
BlobDiffReducer<Dtype>::BlobDiffReducer(NetThread<Dtype> *net_thread):
net_thread_(net_thread){
	sb_.reset(new StreamBroadcast<Dtype>);
	sb_->Init(Caffe::GetActiveDevices());
}

// sequentially reduce gradients
template<typename Dtype>
void BlobDiffReducer<Dtype>::ReduceGpuDiff(
		std::map<int, shared_ptr<Blob<Dtype> > > &shards, Dtype diff_scale) {
	NetThread<Dtype>* net_thread = net_thread_;
	// device_id -> blob
	std::map<int, shared_ptr<Blob<Dtype> > > blobs;
	Net<Dtype>* net = net_thread->get_net();
	int device_id = net_thread->get_device_id();
	const std::vector<int>& device_ids = Caffe::GetActiveDevices();

	blobs[device_id] = shards[device_id];
	for (int i = 0; i < device_ids.size(); ++i) {
		int device_id2 = device_ids[i];
		if (device_id2 != device_id) {
			blobs[device_id2] = shards[device_id2];
			sb_->TransferGpuDiff(blobs, device_id2, (Dtype) 1.0,
					diff_scale * net->GetBatchSizeRatio(device_id2));
			blobs.erase(device_id2);
		}
	}
}

INSTANTIATE_CLASS(BlobDiffReducer);

} // namespace caffe
