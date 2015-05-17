#ifndef CAFFE_IMAGE_ENHANCEMENT_DATA_TRANSFORMER_HPP
#define CAFFE_IMAGE_ENHANCEMENT_DATA_TRANSFORMER_HPP


#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

namespace caffe {

template <typename Dtype>
class ImageEnhancementDataTransformer{
public:
 explicit ImageEnhancementDataTransformer(const ImageEnhancementTransformationParameter& param, Phase phase);
 virtual ~ImageEnhancementDataTransformer() {}

 void Transform(ImageEnhancementDatum& datum, shared_ptr<db::Transaction>& global_ftr_transaction,
		 Blob<Dtype>* transformed_global_ftr,
		 Blob<Dtype>* transformed_semantic_context_ftr, Blob<Dtype>* transformed_pixel_ftr);


protected:
 ImageEnhancementTransformationParameter param_;
 Phase phase_;
 Blob<Dtype> global_ftr_mean_;
 Blob<Dtype> semantic_context_ftr_mean_;
 Blob<Dtype> pixel_ftr_mean_;

};

} // namespace caffe

#endif // CAFFE_IMAGE_ENHANCEMENT_DATA_TRANSFORMER_HPP
