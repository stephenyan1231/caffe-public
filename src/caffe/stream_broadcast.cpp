#include "caffe/stream_broadcast.hpp"
#include "caffe/blob.hpp"

namespace caffe {

template<typename Dtype>
void StreamBroadcast<Dtype>::TransferGpuDiff(
		std::map<int, shared_ptr<Blob<Dtype> > > &blobs, int src_device,
		Dtype scale_tgt, Dtype scale_src) {
	int old_device = Caffe::GetDeviceId();
	CHECK_GT(blobs.count(src_device), 0);
//	Init(blobs);
	if (blobs.size() > 1) {
		if (blobs[src_device]->count() == 0) {
			for (typename std::map<int, shared_ptr<Blob<Dtype> > >::const_iterator it =
					blobs.begin(); it != blobs.end(); ++it) {
				it->second->ReshapeLike(*blobs[src_device].get());
			}
		} else {
			int tgt_device =
					blobs.begin()->first == src_device ?
							(++blobs.begin())->first : blobs.begin()->first;
			DLOG(INFO)<<"StreamBroadcast<Dtype>::TransferGpuDiff";
			if (blobs.size() == 2 && Caffe::CanAccessPeer(src_device, tgt_device)) {
				// a simple P2P copy
				Caffe::SetDevice(tgt_device);
//				Caffe::CublasSetStream(Caffe::cublas_handle(), Caffe::GetDefaultStream());
//				Caffe::CublasSetStream(Caffe::cublas_handle(), streams_[tgt_device]);
				nvtxMarkA("StreamBroadcast<Dtype>::TransferGpuDiff m0");
				caffe_gpu_axpby<Dtype>(blobs[src_device]->count(), scale_src,
						blobs[src_device]->gpu_diff(), scale_tgt,
						blobs[tgt_device]->mutable_gpu_diff());
				nvtxMarkA("StreamBroadcast<Dtype>::TransferGpuDiff m1");
//				Caffe::SyncStream(streams_[tgt_device]);
//				Caffe::SyncStream();
//				Caffe::CublasSetStream(Caffe::cublas_handle());
//				Caffe::SyncDevice();
			} else {
				// TO DO
				// w/o P2P case
			}

		}
	}
	Caffe::SetDevice(old_device);
}

template<typename Dtype>
void StreamBroadcast<Dtype>::Init(const std::vector<int>& device_ids) {
	int old_device = Caffe::GetDeviceId();
	for (int i = 0; i < device_ids.size(); ++i) {
		if (streams_.count(device_ids[i]) == 0) {
			Caffe::Get().SetDevice(device_ids[i]);
			CUDA_CHECK(
					cudaStreamCreateWithFlags(&streams_[device_ids[i]],
							cudaStreamNonBlocking));
		}
	}
	Caffe::SetDevice(old_device);
}

INSTANTIATE_CLASS(StreamBroadcast);
} // namespace caffe
