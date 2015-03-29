#include <opencv2/core/core.hpp>

#include <stdint.h>

#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/data_variable_size_layer.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/data_manager.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<typename Dtype>
DataVariableSizeLayer<Dtype>::~DataVariableSizeLayer<Dtype>() {
}

template<typename Dtype>
void DataVariableSizeLayer<Dtype>::DataLayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
//	int num_replicas = Caffe::GetReplicasNum();
//	int replica_batch_size = divide_up(
//			this->layer_param_.data_variable_size_param().batch_size(), num_replicas);
//	int rest_size = this->layer_param_.data_variable_size_param().batch_size()
//			- this->replica_id_ * replica_batch_size;
//	int this_replica_batch_size = std::min(replica_batch_size, rest_size);
//	this->net_->SetBatchSize(this->replica_id_, this_replica_batch_size);

	int this_replica_batch_size = this->net_->GetBatchSize(this->replica_id_);

	DataVariableSizeManager<Dtype> *dm = dynamic_cast<DataVariableSizeManager<
			Dtype>*>(this->net_->GetDataManager());


	int datum_channels = dm->GetDatumChannels();
	LOG(INFO)<<"DataVariableSizeLayer<Dtype>::DataLayerSetUp datum_channels "<<datum_channels;
//	int datum_height = dm->GetDatumMaxHeight();
//	int datum_width = dm->GetDatumMaxWidth();
//	// image
//
	top[0]->Reshape(this_replica_batch_size, datum_channels, 0,
			0);
//	LOG(INFO)<< "output data size: " << top[0]->num() << ","
//	<< top[0]->channels() << "," << top[0]->height() << ","
//	<< top[0]->width();

	top[1]->Reshape(this_replica_batch_size, 1, 1, 2);
	// label
	if (this->output_labels_) {
		top[2]->Reshape(this_replica_batch_size, 1, 1, 1);
	}
}

template<typename Dtype>
void DataVariableSizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
  			const vector<Blob<Dtype>*>& top){

}

template<typename Dtype>
void DataVariableSizeLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	BaseDataManager<Dtype> *dm = this->net_->GetDataManager();
	dm->CopyFetchDataToConvThread(this->replica_id_, top);
}

INSTANTIATE_CLASS(DataVariableSizeLayer);
REGISTER_LAYER_CLASS(DataVariableSize);

}  // namespace caffe
