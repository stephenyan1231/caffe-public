// Copyright 2014 Zhicheng Yan@eBay.

#include <stdint.h>
#include <leveldb/db.h>
#include <pthread.h>

#include <string>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/proto/caffe.pb.h"

using std::string;

int trans_max = 6;

namespace caffe {

template<typename Dtype>
void* FloatDataTranslateLayerPrefetch(void* layer_pointer) {
	CHECK(layer_pointer);
	FloatDataTranslateLayer<Dtype>* layer =
			static_cast<FloatDataTranslateLayer<Dtype>*>(layer_pointer);
	CHECK(layer);
	Datum datum;
	CHECK(layer->prefetch_data_);
	Dtype* top_data = layer->prefetch_data_->mutable_cpu_data();
	Dtype* top_label;
	if (layer->output_labels_) {
		top_label = layer->prefetch_label_->mutable_cpu_data();
	}
	const Dtype scale = layer->layer_param_.data_param().scale();
	const int batch_size = layer->layer_param_.data_param().batch_size();
	const int crop_size = layer->layer_param_.data_param().crop_size();
	const bool mirror = layer->layer_param_.data_param().mirror();

	if (mirror && crop_size == 0) {
		LOG(FATAL)
				<< "Current implementation requires mirror and crop_size to be "
				<< "set at the same time.";
	}
	// datum scales
	const int channels = layer->datum_channels_;
	const int height = layer->datum_height_;
	const int width = layer->datum_width_;
	const int size = layer->datum_size_;
	const Dtype* mean = layer->data_mean_.cpu_data();
	for (int item_id = 0; item_id < batch_size; ++item_id) {
		// get a blob
		switch (layer->layer_param_.data_param().backend()) {
		case DataParameter_DB_LEVELDB:
			CHECK(layer->iter_);
			CHECK(layer->iter_->Valid());
			datum.ParseFromString(layer->iter_->value().ToString());
			break;
		case DataParameter_DB_LMDB:
			CHECK_EQ(
					mdb_cursor_get(layer->mdb_cursor_, &layer->mdb_key_,
							&layer->mdb_value_, MDB_GET_CURRENT), MDB_SUCCESS);
			datum.ParseFromArray(layer->mdb_value_.mv_data,
					layer->mdb_value_.mv_size);
			break;
		default:
			LOG(FATAL) << "Unknown database backend";
		}

		if (crop_size) {
			CHECK(datum.float_data_size());
			int h_off, w_off;
			int h_trans, w_trans;
			// We only do random crop when we do training.
			if (layer->phase_ == Caffe::TRAIN) {
				h_off = layer->PrefetchRand() % (height - crop_size);
				w_off = layer->PrefetchRand() % (width - crop_size);
				h_trans = (layer->PrefetchRand() % (trans_max)) * ((layer->PrefetchRand()%2)*2-1);
				w_trans = (layer->PrefetchRand() % (trans_max)) * ((layer->PrefetchRand()%2)*2-1);
			} else {
				h_off = (height - crop_size) / 2;
				w_off = (width - crop_size) / 2;
				h_trans = 0;
				w_trans = 0;
			}
			if (mirror && layer->PrefetchRand() % 2) {
				// Copy mirrored version
				for (int c = 0; c < channels; ++c) {
					for (int h = 0; h < crop_size; ++h) {
						for (int w = 0; w < crop_size; ++w) {
							int top_index = ((item_id * channels + c)
									* crop_size + h) * crop_size
									+ (crop_size - 1 - w);
							int h2 = h + h_off + h_trans;
							if(h2 < 0)
								h2 = -h2;
							else if(h2 >= height)
								h2 = 2 * (height - 1) - h2;
							int w2 = w + w_off + w_trans;
							if(w2<0)
								w2=-w2;
							else if(w2>=width)
								w2=2*(width-1)-w2;

							int data_index = (c * height + h2) * width
									+ w2;
							Dtype datum_element =
									static_cast<Dtype>((datum.float_data(
											data_index)));
							top_data[top_index] = (datum_element
									- mean[data_index]) * scale;
						}
					}
				}
//				int top_index = item_id * channels * crop_size * crop_size;
//				LOG(INFO)<<"top_data [top_index] "<<top_data[top_index];
			} else {
				// Normal copy
				for (int c = 0; c < channels; ++c) {
					for (int h = 0; h < crop_size; ++h) {
						for (int w = 0; w < crop_size; ++w) {
							int top_index = ((item_id * channels + c)
									* crop_size + h) * crop_size + w;
							int h2 = h + h_off + h_trans;
							if(h2 < 0)
								h2 = -h2;
							else if(h2 >= height)
								h2 = 2 * (height - 1) - h2;
							int w2 = w + w_off + w_trans;
							if(w2<0)
								w2=-w2;
							else if(w2>=width)
								w2=2*(width-1)-w2;
							int data_index = (c * height + h2) * width
									+ w2;
							Dtype datum_element =
									static_cast<Dtype>((datum.float_data(
											data_index)));
							top_data[top_index] = (datum_element
									- mean[data_index]) * scale;
						}
					}
				}
//				int top_index = item_id * channels * crop_size * crop_size;
//				LOG(INFO)<<"top_data [top_index] "<<top_data[top_index];
			}
		} else {
			for (int j = 0; j < size; ++j) {
				top_data[item_id * size + j] = (datum.float_data(j) - mean[j])
						* scale;
			}

		}

		if (layer->output_labels_) {
			top_label[item_id] = datum.label();
//			LOG(INFO)<<"top_label[item_id] "<<top_label[item_id];
		}
		// go to the next iter
		switch (layer->layer_param_.data_param().backend()) {
		case DataParameter_DB_LEVELDB:
			layer->iter_->Next();
			if (!layer->iter_->Valid()) {
				// We have reached the end. Restart from the first.
				DLOG(INFO) << "Restarting data prefetching from start.";
				layer->iter_->SeekToFirst();
			}
			break;
		case DataParameter_DB_LMDB:
			if (mdb_cursor_get(layer->mdb_cursor_, &layer->mdb_key_,
					&layer->mdb_value_, MDB_NEXT) != MDB_SUCCESS) {
				// We have reached the end. Restart from the first.
				DLOG(INFO) << "Restarting data prefetching from start.";
				CHECK_EQ(
						mdb_cursor_get(layer->mdb_cursor_, &layer->mdb_key_,
								&layer->mdb_value_, MDB_FIRST), MDB_SUCCESS);
			}
			break;
		default:
			LOG(FATAL) << "Unknown database backend";
		}
	}

	return static_cast<void*>(NULL);
}

template<typename Dtype>
void FloatDataTranslateLayer<Dtype>::CreatePrefetchThread() {
	DataLayer<Dtype>::phase_ = Caffe::phase();
	const bool prefetch_needs_rand = (DataLayer<Dtype>::phase_ == Caffe::TRAIN)
			&& (this->layer_param_.data_param().mirror()
					|| this->layer_param_.data_param().crop_size());
	if (prefetch_needs_rand) {
		const unsigned int prefetch_rng_seed = caffe_rng_rand();
		DataLayer<Dtype>::prefetch_rng_.reset(
				new Caffe::RNG(prefetch_rng_seed));
	} else {
		DataLayer<Dtype>::prefetch_rng_.reset();
	}
	// Create the thread.
	CHECK(
			!pthread_create(&(this->thread_), NULL,
					FloatDataTranslateLayerPrefetch<Dtype>, static_cast<void*>(this)))
			<< "Pthread execution failed.";
}

INSTANTIATE_CLASS(FloatDataTranslateLayer);

}  // namespace caffe
