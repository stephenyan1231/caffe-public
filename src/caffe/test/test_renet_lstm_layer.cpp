#include <cmath>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/renet_lstm_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template<typename TypeParam>
class ReNetLSTMLayerTest: public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  ReNetLSTMLayerTest() :
      blob_bottom_(new Blob<Dtype>(2, 1, 4, 4)), blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~ReNetLSTMLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  virtual void InitParam(ReNetLSTMParameter *renet_lstm_param,
      ReNetLSTMParameter::Direction dir,
      int num_output,
      int patch_width, int patch_height) {
    renet_lstm_param->set_direction(dir);
    renet_lstm_param->set_num_output(num_output);
    renet_lstm_param->set_patch_width(patch_width);
    renet_lstm_param->set_patch_height(patch_height);

    renet_lstm_param->mutable_general_weight_filler()->set_type("uniform");
    renet_lstm_param->mutable_general_weight_filler()->set_min(-1);
    renet_lstm_param->mutable_general_weight_filler()->set_max(1);

    renet_lstm_param->mutable_general_bias_filler()->set_type("constant");
    renet_lstm_param->mutable_general_bias_filler()->set_value(0);

    renet_lstm_param->mutable_forget_gate_bias_filler()->set_type("constant");
    renet_lstm_param->mutable_forget_gate_bias_filler()->set_value(1.0);
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ReNetLSTMLayerTest, TestDtypesAndDevices);

TYPED_TEST(ReNetLSTMLayerTest, TestGradientXdirPatch1x1) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ReNetLSTMParameter *renet_lstm_param = layer_param.mutable_renet_lstm_param();
  this->InitParam(renet_lstm_param, ReNetLSTMParameter_Direction_X_DIR, 2, 1, 1);

  ReNetLSTMLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(ReNetLSTMLayerTest, TestGradientXdirPatch2x2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ReNetLSTMParameter *renet_lstm_param = layer_param.mutable_renet_lstm_param();
  this->InitParam(renet_lstm_param, ReNetLSTMParameter_Direction_X_DIR, 2, 2, 2);

  ReNetLSTMLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}


TYPED_TEST(ReNetLSTMLayerTest, TestGradientYdirPatch1x1) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ReNetLSTMParameter *renet_lstm_param = layer_param.mutable_renet_lstm_param();
  this->InitParam(renet_lstm_param, ReNetLSTMParameter_Direction_Y_DIR, 2, 1, 1);

  ReNetLSTMLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(ReNetLSTMLayerTest, TestGradientYdirPatch2x2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ReNetLSTMParameter *renet_lstm_param = layer_param.mutable_renet_lstm_param();
  this->InitParam(renet_lstm_param, ReNetLSTMParameter_Direction_Y_DIR, 2, 2, 2);

  ReNetLSTMLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
}

}  // namespace caffe
