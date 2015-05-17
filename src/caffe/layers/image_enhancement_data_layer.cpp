#include "caffe/net.hpp"
#include "caffe/image_enhancement_data_layer.hpp"
#include "caffe/data_manager.hpp"

namespace caffe {


template<typename Dtype>
ImageEnhancementDataLayer<Dtype>::~ImageEnhancementDataLayer(){

}

template<typename Dtype>
void ImageEnhancementDataLayer<Dtype>::DataLayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	int this_replica_batch_size = this->net_->GetBatchSize(this->replica_id_);

	ImageEnhancementDataManager<Dtype>* data_manager =
			dynamic_cast<ImageEnhancementDataManager<Dtype>*>(this->net_->GetDataManager());

	top[0]->Reshape(this_replica_batch_size, data_manager->get_global_ftr_dim(),
			1, 1);
	top[1]->Reshape(this_replica_batch_size, data_manager->get_semantic_context_ftr_dim(),
			1, 1);
	top[2]->Reshape(this_replica_batch_size, data_manager->get_pixel_ftr_dim(),
			1, 1);
	top[3]->Reshape(this_replica_batch_size,
			data_manager->get_pixel_samples_num_per_segment() * data_manager->get_color_basis_dim(),
			1, 1);
	top[4]->Reshape(this_replica_batch_size,
			data_manager->get_pixel_samples_num_per_segment() * data_manager->get_color_dim(),
			1, 1);
}

template<typename Dtype>
void ImageEnhancementDataLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	this->net_->GetDataManager()->CopyFetchDataToConvThread(this->replica_id_, top);
}

INSTANTIATE_CLASS(ImageEnhancementDataLayer);
REGISTER_LAYER_CLASS(ImageEnhancementData);

} // namespace caffe
