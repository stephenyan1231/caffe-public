#ifndef CAFFE_SPATIAL_PROB_AGGREGATOIN_LAYER_HPP_
#define CAFFE_SPATIAL_PROB_AGGREGATOIN_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/neuron_layers.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {

/**
 * @brief Computes the classification accuracy for a one-of-many
 *        classification task.
 *        The final classification feature map has spatial height and width dimensions
 *        thus its size is (num_image, num_classes, height, width)
 *        there is still one label for each image
 */
template <typename Dtype>
class SpatialProbAggregationLayer : public Layer<Dtype> {
 public:
  explicit SpatialProbAggregationLayer(const LayerParameter& param, int replica_id, Net<Dtype> *net)
      : Layer<Dtype>(param,replica_id,net) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "SpatialProbAggregation"; }
  /* 1st bottom blob: spatial probability
   * 2nd bottom blob: label
   * 3rd bottom blob: image variable (height,width) (optional)
   * */
  virtual inline int ExactNumBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    for (int i = 0; i < propagate_down.size(); ++i) {
      if (propagate_down[i]) { NOT_IMPLEMENTED; }
    }
  }

  int top_k_;
  int spatial_subsample_height_factor_;
  int spatial_subsample_width_factor_;

};

} // namespace caffe

#endif
