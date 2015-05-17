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
	int this_replica_batch_size = this->net_->GetBatchSize(this->replica_id_);
	DataVariableSizeManager<Dtype> *dm = dynamic_cast<DataVariableSizeManager<
			Dtype>*>(this->net_->GetDataManager());

	int datum_channels = dm->GetDatumChannels();
	top[0]->Reshape(this_replica_batch_size, datum_channels, 0, 0);
}

template<typename Dtype>
void DataVariableSizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	int this_replica_batch_size = this->net_->GetBatchSize(this->replica_id_);
	top[1]->Reshape(this_replica_batch_size, 1, 1, 2);
	// label
	if (this->output_labels_) {
		top[2]->Reshape(this_replica_batch_size, 1, 1, 1);
	}
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
