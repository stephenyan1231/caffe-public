#ifndef CAFFE_STREAM_BROADCAST_HPP_
#define CAFFE_STREAM_BROADCAST_HPP_

#include "caffe/common.hpp"

#include <map>
#include <vector>

namespace caffe {

template <typename Dtype>
class Blob;

template <typename Dtype>
class StreamBroadcast{
public:
	explicit StreamBroadcast(){

	}

	void Init(const std::vector<int>& device_ids);
	void TransferGpuDiff(std::map<int, shared_ptr<Blob<Dtype> > > &blobs,
			int src_device, Dtype scale_tgt, Dtype scale_src);

protected:

	// device_id -> cuda stream
	std::map<int, cudaStream_t> streams_;

  DISABLE_COPY_AND_ASSIGN(StreamBroadcast);
}; // class StreamBroadCast


} // namespace caffe

#endif // CAFFE_STREAM_BROADCAST_HPP_
