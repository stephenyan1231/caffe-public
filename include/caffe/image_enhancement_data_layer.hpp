#ifndef CAFFE_IMAGE_ENHANCEMENT_DATA_LAYER_HPP_
#define CAFFE_IMAGE_ENHANCEMENT_DATA_LAYER_HPP_

#include "caffe/data_layers.hpp"

namespace caffe {

template <typename Dtype>
class ImageEnhancementDataLayer : public BaseDataLayer<Dtype> {
public:
 explicit ImageEnhancementDataLayer(const LayerParameter& param, int replica_id, Net<Dtype> *net)
     : BaseDataLayer<Dtype>(param,replica_id,net) {}
 virtual ~ImageEnhancementDataLayer();
 virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top);

 virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top);

 virtual inline const char* type() const { return "ImageEnhancementData"; }
 virtual inline int ExactNumBottomBlobs() const { return 0; }
 /* 1st output: image global feature
  * 2nd output: semantic context feature
  * 3rd output: pixel feature
  * 4th output: quadratic color basis
  * 5th output: ground truth CIELAB color
 */
 virtual inline int ExactNumTopBlobs() const { return 5; }

protected:
};

} // namespace caffe


#endif //CAFFE_IMAGE_ENHANCEMENT_DATA_LAYER_HPP_
