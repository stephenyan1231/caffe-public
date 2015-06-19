#ifndef CAFFE_SEQUENCE_2D_LAYERS_HPP_
#define CAFFE_SEQUENCE_2D_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/*
 * @brief An abstract class for implementing 2D recurrent/LSTM neural network.
 * Currently, an instantiation LSTM2DLayer based on LSTM is implemented.
 * */

template<typename Dtype>
class Recurrent2DLayer: public Layer<Dtype> {
public:
  explicit Recurrent2DLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Recurrent2D"; }
  /* @brief Input is a blob of shape (c, h, w)
   * */
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  /* @brief Output a single blob consisting of 4 stacked layers of hidden states
   * */
  virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
/* @brief Fills net_param with the recurrent network architecture.
 *
 * */
  virtual void FillUnrolledNet(NetParameter* net_param) const = 0;

  /**
   * @brief Fills names with the names of the 0th timestep recurrent input
   *        Blob&s.  Subclasses should define this -- see RNNLayer and LSTMLayer
   *        for examples.
   */
  virtual void RecurrentInputBlobNames(vector<string>* names) const = 0;

  /**
   * @brief Fills names with the names of the output blobs, concatenated across
   *        all patches.  Should return a name for each top Blob.
   *        Subclasses should define this -- see LSTM2DLayer for
   *        examples.
   */
  virtual void OutputBlobNames(vector<string>* names) const = 0;


  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  /// @brief A helper function, useful for stringifying indices (r,c).
  virtual string coordinate_to_str(const int r, const int c) const;

  /// @brief A helper function, useful for stringifying scanning direction i.
  virtual string direction_to_str(const int i) const;

  /// @brief prepare padded input images
//  virtual void pad_image(const Blob<Dtype>* image);

  /// @brief prepare a grid of patches with padding for each scanning direction
  virtual void prepare_patch_with_padding(const Blob<Dtype>* image);

  /// @brief A Net to implement the Recurrent functionality.
  shared_ptr<Net<Dtype> > unrolled_net_;

  /// @brief input image channels
  int num_, channels_;

  /// @brief patch width and height
  int patch_w_, patch_h_;

  /// @brief number of patches in x and y direction
  int patch_nx_, patch_ny_;

  vector<Blob<Dtype>* > output_blobs_;

  // four padded input images for four scanning directions
  vector<Blob<Dtype>* > x_input_blobs_;

//  vector<shared_ptr<Blob<Dtype> > > input_blobs_;
};

template <typename Dtype>
class LSTM2DLayer:public Recurrent2DLayer<Dtype> {
public:
	explicit LSTM2DLayer(const LayerParameter& param)
  : Recurrent2DLayer<Dtype>(param) {}

  virtual inline const char* type() const { return "LSTM2D"; }

protected:
  virtual void FillUnrolledNet(NetParameter* net_param) const;
  virtual void RecurrentInputBlobNames(vector<string>* names) const;
  virtual void OutputBlobNames(vector<string>* names) const;
};

template <typename Dtype>
class LSTM2DUnitLayer:public Layer<Dtype> {
public:
	explicit LSTM2DUnitLayer(const LayerParameter& param)
  : Layer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "LSTM2DUnit"; }
  /// @brief three inputs. Take one scanning direction as the example
  //	1) e.g. c_i_{j-1}
  //	2) e.g. c_{i-1}_j
  //	3) e.g. gate^{pp}_input_i_j
  virtual inline int ExactNumBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  // mini-batch size
  int num_;

  /// @brief the hidden/output dimension
  int hidden_dim_;
};

} // namespace caffe

#endif // CAFFE_SEQUENCE_2D_LAYERS_HPP_
