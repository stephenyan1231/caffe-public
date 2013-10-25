// Copyright Yangqing Jia 2013

#include <string>
#include <map>

#include "caffe/layer.hpp"
#include "caffe/layer_register.hpp"
#include "caffe/proto/caffe.pb.h"

using std::string;

namespace caffe {

template <typename Dtype>
void LayerRegistry<Dtype>::AddCreator(string name, Creator creator) {
  LOG(INFO) << "Registering Layer " << name;
  layer_map_[name] = creator;
}

template <typename Dtype>
Layer<Dtype>* LayerRegistry<Dtype>::CreateLayer(const string& name,
      const LayerParameter& param) {
  typename LayerMap::const_iterator it = layer_map_.find(name);
  if (it == layer_map_.end()) {
    LOG(ERROR) << "Unknown layer: " << name;
    LOG(ERROR) << "Available Layers:";
    for (typename LayerMap::const_iterator it = layer_map_.begin();
        it != layer_map_.end(); ++it) {
      LOG(ERROR) << "\t" << it->first;
    }
    LOG(FATAL) << "Please double-check your layer registration.";
  }
  return (it->second)(param);
}

INSTANTIATE_CLASS(LayerRegistry);


// Internal: The registerer class to register a class. Oh my god.
template <typename Dtype>
LayerCreatorRegisterer<Dtype>::LayerCreatorRegisterer(const string& name,
    typename LayerRegistry<Dtype>::Creator creator) {
  LayerRegistry<Dtype>::Get()->AddCreator(name, creator);
}

INSTANTIATE_CLASS(LayerCreatorRegisterer);


}  // namespace caffe

