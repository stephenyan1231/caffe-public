// This function implements a registry keeping a record of all layer names.
// Copyright Yangqing Jia 2013

#ifndef CAFFE_LAYER_REGISTER_HPP_
#define CAFFE_LAYER_REGISTER_HPP_

#include <string>
#include <map>

#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

using std::string;

namespace caffe {

// Internal: the layer registry
template <typename Dtype>
class LayerRegistry {
 public:
  typedef Layer<Dtype>* (*Creator)(const LayerParameter&);
  ~LayerRegistry() {}
  static LayerRegistry<Dtype>* Get() {
    if (!layer_registry_singleton_) {
        layer_registry_singleton_ = new LayerRegistry<Dtype>();
    }
    return layer_registry_singleton_;
  }
  void AddCreator(string name, Creator creator);
  Layer<Dtype>* CreateLayer(const string& name, const LayerParameter& param);

 private:
  LayerRegistry() : layer_map_() {}
  typedef typename std::map<string, Creator> LayerMap;
  LayerMap layer_map_;
  static LayerRegistry<Dtype>* layer_registry_singleton_;
};

// The singleton.
template <typename Dtype> LayerRegistry<Dtype>* LayerRegistry<Dtype>::layer_registry_singleton_ = NULL;

// Internal: The registerer class to register a class. Oh my god.
template <typename Dtype>
class LayerCreatorRegisterer {
 public:
  explicit LayerCreatorRegisterer(const string& name,
      typename LayerRegistry<Dtype>::Creator creator);
  ~LayerCreatorRegisterer() {}
};


// The function to call to get a layer.
template <typename Dtype>
Layer<Dtype>* CreateLayer(const LayerParameter& param) {
  return LayerRegistry<Dtype>::Get()->CreateLayer(param.type(), param);
}


// The macro to use for register a layer. For example, if you have a
// ConvolutionLayer and want to register it with name "conv", do
//    REGISTER_LAYER("conv", ConvolutionLayer)
#define REGISTER_LAYER(name, DerivedLayer) \
  template <typename Dtype> \
  Layer<Dtype>* Create##DerivedLayer(const LayerParameter& param) { \
    return new DerivedLayer<Dtype>(param); \
  } \
  LayerCreatorRegisterer<float> g_creator_float_##DerivedLayer( \
      name, &Create##DerivedLayer<float>); \
  LayerCreatorRegisterer<double> g_creator_double_##DerivedLayer( \
      name, &Create##DerivedLayer<double>)


}  // namespace caffe

# endif  // CAFFE_LAYER_REGISTER_HPP_
