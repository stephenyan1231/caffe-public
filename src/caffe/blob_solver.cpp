#include "caffe/blob_solver.hpp"
#include "caffe/blob.hpp"
#include "caffe/blob_diff_reducer.hpp"
#include "caffe/copy_pipeline.hpp"
#include "caffe/net.hpp"
#include "caffe/util/io.hpp"
#include "caffe/solver.hpp"

#include <map>

namespace caffe {

template<typename Dtype>
BlobSolver<Dtype>::BlobSolver(const SolverParameter& param, int param_id,
		NetThread<Dtype>* net_thread) :
		net_thread_(net_thread), param_id_(param_id) {
	Init(param);
}

template<typename Dtype>
BlobSolver<Dtype>::BlobSolver(const string& param_file, int param_id,
		NetThread<Dtype>* net_thread) :
		net_thread_(net_thread), param_id_(param_id) {
	SolverParameter param;
	ReadProtoFromTextFileOrDie(param_file, &param);
	Init(param);
}

template<typename Dtype>
void BlobSolver<Dtype>::Init(const SolverParameter& param) {
	param_ = param;
	CHECK_GE(param_.average_loss(), 1) << "average_loss should be non-negative.";
	if (param_.random_seed() >= 0) {
		Caffe::set_random_seed(param_.random_seed());
	}
}

INSTANTIATE_CLASS(BlobSolver);

template<typename Dtype>
void BlobSGDSolver<Dtype>::PreSolve_() {
	const vector<int>& params_shard_size = this->net_thread_->params_shard_size();
	history_.reset(new Blob<Dtype>(params_shard_size[this->param_id_], 1, 1, 1));
	update_.reset(new Blob<Dtype>(params_shard_size[this->param_id_], 1, 1, 1));
	temp_.reset(new Blob<Dtype>(params_shard_size[this->param_id_], 1, 1, 1));
}

template<typename Dtype>
void BlobSGDSolver<Dtype>::ComputeUpdateValue_() {
	NetThread<Dtype> *net_thread = this->net_thread_;
	const vector<shared_ptr<Blob<Dtype> > >& net_params = net_thread->params();
	const vector<float>& net_params_lr = net_thread->params_lr();
	const vector<float>& net_params_weight_decay =
			net_thread->params_weight_decay();
	std::map<int, NetThread<Dtype>*>& replicas = net_thread->get_replicas();

	int device_id = net_thread->get_device_id();
	int param_id = this->param_id_;

	// get the learning rate
	SGDSolver<Dtype>* solver =
			dynamic_cast<SGDSolver<Dtype> *>(net_thread->GetExternalSolver());
	SolverParameter param = solver->GetParams();
	Dtype rate = solver->GetLearningRate();
	if (param.display() && solver->iter() % param.display() == 0) {
		DLOG(INFO)<< "Iteration " << solver->iter() << ", lr = " << rate;
	}
	Dtype momentum = param.momentum();
	Dtype weight_decay = param.weight_decay();
	string regularization_type = param.regularization_type();

	Dtype local_rate = rate * net_params_lr[param_id];
	Dtype local_decay = weight_decay * net_params_weight_decay[param_id];

	switch (Caffe::mode()) {
	case Caffe::CPU: {
		// Compute the value to history, and then copy them to the blob's diff.
		if (local_decay) {
			if (regularization_type == "L2") {
				// add weight decay
				caffe_axpy(net_params[param_id]->count(), local_decay,
						net_params[param_id]->cpu_data(),
						net_params[param_id]->mutable_cpu_diff());
			} else if (regularization_type == "L1") {
				caffe_cpu_sign(net_params[param_id]->count(),
						net_params[param_id]->cpu_data(), temp_->mutable_cpu_data());
				caffe_axpy(net_params[param_id]->count(), local_decay,
						temp_->cpu_data(), net_params[param_id]->mutable_cpu_diff());
			} else {
				LOG(FATAL)<< "Unknown regularization type: " << regularization_type;
			}
		}
		caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
				net_params[param_id]->cpu_diff(), momentum,
				history_->mutable_cpu_data());
		// copy
		caffe_copy(net_params[param_id]->count(), history_->cpu_data(),
				net_params[param_id]->mutable_cpu_diff());
		break;
	}
	case Caffe::GPU:
	{
#ifndef CPU_ONLY
		// Compute the value to history, and then copy them to the blob's diff.
		// device id -> Blob
		std::map<int, shared_ptr<Blob<Dtype> > > shards;
		for (typename std::map<int, NetThread<Dtype>*>::iterator it =
				replicas.begin(); it != replicas.end(); ++it) {
			shards[it->second->get_device_id()] = it->second->GetShardGPUOnly(param_id,
					net_thread->get_replica_id());
		}
	  caffe_gpu_scal<Dtype>(shards[device_id]->count(),
	  		net_thread->get_net()->GetBatchSizeRatio(device_id),
	  		shards[device_id]->mutable_gpu_diff());
		// Reduce everyone's gradient gpu_diff
		this->get_blob_diff_reducer()->ReduceGpuDiff(shards, (Dtype) 1.0);

		if (local_decay) {
			if (regularization_type == "L2") {
				// add weight decay
				caffe_gpu_axpy<Dtype>(shards[device_id]->count(), local_decay,
						shards[device_id]->gpu_data(),
						shards[device_id]->mutable_gpu_diff());
			} else if (regularization_type == "L1") {
				caffe_gpu_sign<Dtype>(shards[device_id]->count(),
						shards[device_id]->gpu_data(),
						temp_->mutable_gpu_data());
				caffe_gpu_axpy<Dtype>(shards[device_id]->count(), local_decay,
						temp_->gpu_data(),
						shards[device_id]->mutable_gpu_diff());
			} else {
				LOG(FATAL)<< "Unknown regularization type: " << regularization_type;
			}
		}
		caffe_gpu_axpby<Dtype>(shards[device_id]->count(), local_rate,
				shards[device_id]->gpu_diff(), momentum,
				history_->mutable_gpu_data());
		// copy
		caffe_copy<Dtype>(shards[device_id]->count(),
				history_->gpu_data(),
				shards[device_id]->mutable_gpu_diff());
		// crucial. Make sure the gradients are ready before broadcasting them
		// to other replicas
		Caffe::SyncDevice();
		// broadcast gpu diff to everyone
		this->get_blob_diff_broadcaster()->BroadcastGpuDiff(shards,
				(Dtype) 1.0, (Dtype) 0.0);
#else
		NO_GPU;
#endif
		break;
	}
	default: {
		LOG(FATAL)<< "Unknown caffe mode: " << Caffe::mode();
	}
}

}

INSTANTIATE_CLASS(BlobSGDSolver);
}// namespace caffe
