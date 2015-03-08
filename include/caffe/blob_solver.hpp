#ifndef CAFFE_BLOB_SOLVER_HPP_
#define CAFFE_BLOB_SOLVER_HPP_

#include "caffe/common.hpp"
//#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template<typename Dtype>
class Blob;

template<typename Dtype>
class NetThread;

template<typename Dtype>
class BlobDiffReducer;

template <typename Dtype>
class IBroadcastDiffNetwork;

template<typename Dtype>
class BlobSolver {
public:
	explicit BlobSolver(const SolverParameter& param, int param_id, NetThread<Dtype>* net_thread);
	explicit BlobSolver(const string& param_file, int param_id, NetThread<Dtype>* net_thread);
	void Init(const SolverParameter& param);

	virtual void PreSolve() {
		PreSolve_();
	}
	virtual void ComputeUpdateValue() {
		ComputeUpdateValue_();
	}

	shared_ptr<BlobDiffReducer<Dtype> > get_blob_diff_reducer() {
		if (!blob_diff_reducer_.get()) {
			blob_diff_reducer_.reset(new BlobDiffReducer<Dtype>(net_thread_));
		}
		return blob_diff_reducer_;
	}

	shared_ptr<IBroadcastDiffNetwork<Dtype> > get_blob_diff_broadcaster() {
		if (!blob_diff_broadcaster_.get()) {
			int device_id = net_thread_->get_device_id();
			blob_diff_broadcaster_.reset(
					IBroadcastDiffNetwork<Dtype>::make(
							net_thread_->get_net()->GetDeviceIds(), device_id));
		}
		return blob_diff_broadcaster_;
	}

protected:
	virtual void PreSolve_() = 0;
	virtual void ComputeUpdateValue_() = 0;

	SolverParameter param_;
	NetThread<Dtype> *net_thread_;
	int param_id_;

//	Blob<Dtype> *blob_;
	shared_ptr<BlobDiffReducer<Dtype> > blob_diff_reducer_;
	shared_ptr<IBroadcastDiffNetwork<Dtype> > blob_diff_broadcaster_;

DISABLE_COPY_AND_ASSIGN(BlobSolver);
};

template<typename Dtype>
class BlobSGDSolver: public BlobSolver<Dtype> {
public:
	explicit BlobSGDSolver(const SolverParameter& param, int param_id, NetThread<Dtype>* net_thread) :
			BlobSolver<Dtype>(param, param_id, net_thread) {
	}
	explicit BlobSGDSolver(const string& param_file, int param_id, NetThread<Dtype>* net_thread) :
			BlobSolver<Dtype>(param_file, param_id, net_thread) {
	}

protected:
	void PreSolve_();
	void ComputeUpdateValue_();

	shared_ptr<Blob<Dtype> > history_, update_, temp_;

DISABLE_COPY_AND_ASSIGN(BlobSGDSolver);
};

template<typename Dtype>
BlobSolver<Dtype>* GetBlobSolver(const SolverParameter& param, int param_id, NetThread<Dtype>* net_thread) {
	SolverParameter_SolverType type = param.solver_type();
	switch (type) {
	case SolverParameter_SolverType_SGD:
		return new BlobSGDSolver<Dtype>(param, param_id, net_thread);
//  case SolverParameter_SolverType_NESTEROV:
//      return new NesterovSolver<Dtype>(param);
//  case SolverParameter_SolverType_ADAGRAD:
//      return new AdaGradSolver<Dtype>(param);
	default:
		LOG(FATAL)<< "Unknown SolverType: " << type;
	}
	return (BlobSolver<Dtype>*) NULL;
}

} // namespace caffe

#endif // CAFFE_BLOB_SOLVER_HPP_
