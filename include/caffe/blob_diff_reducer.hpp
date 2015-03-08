#ifndef CAFFE_BLOB_DIFF_REDUCER_HPP_
#define CAFFE_BLOB_DIFF_REDUCER_HPP_

#include "caffe/common.hpp"
#include "caffe/stream_broadcast.hpp"

#include <map>

namespace caffe {

template <typename Dtype>
class NetThread;

template <typename Dtype>
class BlobDiffReducer {
 public:
	explicit BlobDiffReducer(NetThread<Dtype> *net_thread):
	net_thread_(net_thread){
		sb_.reset(new StreamBroadcast<Dtype>);
		sb_->Init(net_thread_->get_net()->GetDeviceIds());
	}
	void ReduceGpuDiff(std::map<int, shared_ptr<Blob<Dtype> > > &shards, Dtype diff_scale);

 protected:
	NetThread<Dtype> *net_thread_;

	shared_ptr<StreamBroadcast<Dtype> > sb_;


  DISABLE_COPY_AND_ASSIGN(BlobDiffReducer);
}; // class BlobDiffReducer

} //namespace caffe

#endif // CAFFE_BLOB_DIFF_REDUCER_HPP_
