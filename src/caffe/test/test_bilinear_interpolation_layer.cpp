#include <cmath>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/interpolation_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template<typename TypeParam>
class BilinearInterpolationLayerTest: public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  BilinearInterpolationLayerTest() :
      blob_bottom_(new Blob<Dtype>(2, 2, 3, 3)), blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~BilinearInterpolationLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  virtual void InitParam(BilinearInterpolationParameter *bilinear_interpolation_param,
      int interpolation_factor) {
    bilinear_interpolation_param->set_interpolation_factor(interpolation_factor);
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(BilinearInterpolationLayerTest, TestDtypesAndDevices);

TYPED_TEST(BilinearInterpolationLayerTest, TestGradient2X){
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BilinearInterpolationParameter *bilinear_interpolation_param = layer_param.mutable_bilinear_interpolation_param();
  this->InitParam(bilinear_interpolation_param, 2);

  BilinearInterpolationLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(BilinearInterpolationLayerTest, TestGradient3X){
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BilinearInterpolationParameter *bilinear_interpolation_param = layer_param.mutable_bilinear_interpolation_param();
  this->InitParam(bilinear_interpolation_param, 3);

  BilinearInterpolationLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(BilinearInterpolationLayerTest, TestGradient5X){
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BilinearInterpolationParameter *bilinear_interpolation_param = layer_param.mutable_bilinear_interpolation_param();
  this->InitParam(bilinear_interpolation_param, 5);

  BilinearInterpolationLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe
