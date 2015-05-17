// Copyright 2015 Zhicheng Yan

#ifndef CAFFE_VIDEO_ENHANCEMENT_LAYERS_HPP_
#define CAFFE_VIDEO_ENHANCEMENT_LAYERS_HPP_

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"

namespace caffe {



/* ImageEnhancementSquareDiffLossLayer
*/
template <typename Dtype>
class ImageEnhancementSquareDiffLossLayer : public Layer<Dtype> {
 public:
  explicit ImageEnhancementSquareDiffLossLayer(const LayerParameter& param,
  		int replica_id, Net<Dtype> *net)
      : Layer<Dtype>(param, replica_id, net) {}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
				const vector<Blob<Dtype>*>& top);
  virtual void Reshape(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ImageEnhancementSquareDiffLoss"; }

  /*
   * bottom 0: regresses (3*10)-D coefficients
   * bottom 1: (pixel_samples_num_per_segment_ * 10)-D quadratic color basis
   * bottom 2: ground truth 3-D CIELAB color
   * */
  virtual inline int ExactNumBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  Blob<Dtype> pred_LAB_color_;
  int pixel_samples_num_per_segment_;
	static const int color_basis_dim_ = 10;
	static const int color_dim_ = 3;
};

} // namespace caffe

#endif // CAFFE_VIDEO_ENHANCEMENT_LAYERS_HPP_
