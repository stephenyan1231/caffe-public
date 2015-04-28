#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/shift_pooling_layer.hpp"

namespace caffe {

template<typename Dtype>
__global__ void ShiftMaxPoolForward(const int nthreads,
		const Dtype* bottom_data, const int num, const int channels,
		const int height, const int width, const int pooled_height,
		const int pooled_width, const int kernel_h, const int kernel_w,
		const int stride_h, const int stride_w, const int pad_h, const int pad_w,
		Dtype* top_data, int* mask, Dtype* top_mask) {
	CUDA_KERNEL_LOOP(index, nthreads)
	{
		int pw = index % pooled_width;
		int ph = (index / pooled_width) % pooled_height;
		int c = (index / pooled_width / pooled_height) % channels;
		int sx = (index / pooled_width / pooled_height / channels) % stride_w;
		int sy = (index / pooled_width / pooled_height / channels / stride_w)
				% stride_h;
		int n = index / pooled_width / pooled_height / channels / stride_w
				/ stride_h;
		int hstart = ph * stride_h - pad_h + sy;
		int wstart = pw * stride_w - pad_w + sx;
		int hend = min(hstart + kernel_h, height);
		int wend = min(wstart + kernel_w, width);
		hstart = max(hstart, 0);
		wstart = max(wstart, 0);
		Dtype maxval = -FLT_MAX;
		int maxidx = -1;
		bottom_data += (n * channels + c) * height * width;
		for (int h = hstart; h < hend; ++h) {
			for (int w = wstart; w < wend; ++w) {
				if (bottom_data[h * width + w] > maxval) {
					maxidx = h * width + w;
					maxval = bottom_data[maxidx];
				}
			}
		}
		if(maxidx == -1){
			top_data[index] = 0;
		} else {
			top_data[index] = maxval;
		}
		if (mask) {
			mask[index] = -1;
		} else {
			top_mask[index] = -1;
		}
	}
}

template<typename Dtype>
__global__ void ShiftAvePoolForward(const int nthreads,
		const Dtype* bottom_data, const int num, const int channels,
		const int height, const int width, const int pooled_height,
		const int pooled_width, const int kernel_h, const int kernel_w,
		const int stride_h, const int stride_w, const int pad_h, const int pad_w,
		Dtype* top_data) {
	CUDA_KERNEL_LOOP(index, nthreads)
	{
		int pw = index % pooled_width;
		int ph = (index / pooled_width) % pooled_height;
		int c = (index / pooled_width / pooled_height) % channels;
		int sx = (index / pooled_width / pooled_height / channels) % stride_w;
		int sy = (index / pooled_width / pooled_height / channels / stride_w)
				% stride_h;
		int n = index / pooled_width / pooled_height / channels / stride_w
				/ stride_h;
		int hstart = ph * stride_h - pad_h + sy;
		int wstart = pw * stride_w - pad_w + sx;
		int hend = min(hstart + kernel_h, height + pad_h);
		int wend = min(wstart + kernel_w, width + pad_w);
		int pool_size = (hend - hstart) * (wend - wstart);
		hstart = max(hstart, 0);
		wstart = max(wstart, 0);
		hend = min(hend, height);
		wend = min(wend, width);
		Dtype aveval = 0;
		bottom_data += (n * channels + c) * height * width;
		for (int h = hstart; h < hend; ++h) {
			for (int w = wstart; w < wend; ++w) {
				aveval += bottom_data[h * width + w];
			}
		}
		top_data[index] = aveval / pool_size;
	}
}

template<typename Dtype>
void ShiftPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	nvtxMarkA("ShiftPoolingLayer<Dtype>::Forward_gpu");

	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	int count = top[0]->count();
	// We'll output the mask to top[1] if it's of size >1.
	const bool use_top_mask = top.size() > 1;
	int* mask = NULL;
	Dtype* top_mask = NULL;
	switch (this->layer_param_.pooling_param().pool()) {
	case PoolingParameter_PoolMethod_MAX:
		if (use_top_mask) {
			top_mask = top[1]->mutable_gpu_data();
		} else {
			mask = max_idx_.mutable_gpu_data();
		}
		ShiftMaxPoolForward<Dtype> <<<CAFFE_GET_BLOCKS(count),
				CAFFE_CUDA_NUM_THREADS>>>(count, bottom_data, bottom[0]->num(),
				channels_, height_, width_, pooled_height_, pooled_width_, kernel_h_,
				kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data, mask,
				top_mask);
		break;
	case PoolingParameter_PoolMethod_AVE:
		ShiftAvePoolForward<Dtype> <<<CAFFE_GET_BLOCKS(count),
				CAFFE_CUDA_NUM_THREADS>>>(count, bottom_data, bottom[0]->num(),
				channels_, height_, width_, pooled_height_, pooled_width_, kernel_h_,
				kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data);
		break;
	case PoolingParameter_PoolMethod_STOCHASTIC:
		NOT_IMPLEMENTED;
		break;
	default:
		LOG(FATAL)<< "Unknown pooling method.";
	}

	CUDA_POST_KERNEL_CHECK;
	if (Caffe::phase() == Caffe::TEST && this->conserve_gpu_memory_test_) {
//		LOG(INFO)<<"shift pooling layer name "<<this->layer_param_.name()
//				<<" release bottom blob count "<<bottom[0]->count();
		bottom[0]->ReshapeForceMemoryFree(0, 0, 0, 0);
		if(this->layer_param_.pooling_param().pool() == PoolingParameter_PoolMethod_MAX){
			max_idx_.ReshapeForceMemoryFree(0, 0, 0, 0);
		}
//		LOG(INFO)<<"ShiftPoolingLayer<Dtype>::Forward_gpu free memory";
	}
}

template<typename Dtype>
__global__ void ShiftMaxPoolBackward(const int nthreads, const Dtype* top_diff,
		const int* mask, const Dtype* top_mask, const int num, const int channels,
		const int height, const int width, const int pooled_height,
		const int pooled_width, const int kernel_h, const int kernel_w,
		const int stride_h, const int stride_w, const int pad_h, const int pad_w,
		Dtype* bottom_diff) {
	CUDA_KERNEL_LOOP(index, nthreads)
	{
		// find out the local index
		// find out the local offset
		int w = index % width;
		int h = (index / width) % height;
		int c = (index / width / height) % channels;
		int n = index / width / height / channels;
		Dtype gradient = 0;
		for (int sy = 0; sy < stride_h; ++sy) {
			for (int sx = 0; sx < stride_w; ++sx) {
				int phstart =
						(h - sy + pad_h < kernel_h) ?
								0 : (h - sy + pad_h - kernel_h) / stride_h + 1;
				int phend = min((h - sy + pad_h) / stride_h + 1, pooled_height);
				int pwstart =
						(w - sx + pad_w < kernel_w) ?
								0 : (w - sx + pad_w - kernel_w) / stride_w + 1;
				int pwend = min((w - sx + pad_w) / stride_w + 1, pooled_width);
				int offset = (((n * stride_h + sy) * stride_w + sx) * channels + c)
						* pooled_height * pooled_width;
				top_diff += offset;
				if (mask) {
					mask += offset;
					for (int ph = phstart; ph < phend; ++ph) {
						for (int pw = pwstart; pw < pwend; ++pw) {
							if (mask[ph * pooled_width + pw] == h * width + w) {
								gradient += top_diff[ph * pooled_width + pw];
							}
						}
					}
				} else {
					top_mask += offset;
					for (int ph = phstart; ph < phend; ++ph) {
						for (int pw = pwstart; pw < pwend; ++pw) {
							if (top_mask[ph * pooled_width + pw] == h * width + w) {
								gradient += top_diff[ph * pooled_width + pw];
							}
						}
					}
				}
			}
		}
		bottom_diff[index] = gradient;
	}
}

template<typename Dtype>
__global__ void ShiftAvePoolBackward(const int nthreads, const Dtype* top_diff,
		const int num, const int channels, const int height, const int width,
		const int pooled_height, const int pooled_width, const int kernel_h,
		const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
		const int pad_w, Dtype* bottom_diff) {
	CUDA_KERNEL_LOOP(index, nthreads)
	{
		// find out the local index
		// find out the local offset
		int w = index % width + pad_w;
		int h = (index / width) % height + pad_h;
		int c = (index / width / height) % channels;
		int n = index / width / height / channels;
		Dtype gradient = 0;
		for (int sy = 0; sy < stride_h; ++sy) {
			for (int sx = 0; sx < stride_w; ++sx) {
				int phstart =
						(h - sy < kernel_h) ? 0 : (h - sy - kernel_h) / stride_h + 1;
				int phend = min((h - sy) / stride_h + 1, pooled_height);
				int pwstart =
						(w - sx < kernel_w) ? 0 : (w - sx - kernel_w) / stride_w + 1;
				int pwend = min((w - sx) / stride_w + 1, pooled_width);
				int offset = (((n * stride_h + sy) * stride_w + sx) * channels + c)
						* pooled_height * pooled_width;
				top_diff += offset;
				for (int ph = phstart; ph < phend; ++ph) {
					for (int pw = pwstart; pw < pwend; ++pw) {
						// figure out the pooling size
						int hstart = ph * stride_h - pad_h + sy;
						int wstart = pw * stride_w - pad_w + sx;
						int hend = min(hstart + kernel_h, height + pad_h);
						int wend = min(wstart + kernel_w, width + pad_w);
						int pool_size = (hend - hstart) * (wend - wstart);
						gradient += top_diff[ph * pooled_width + pw] / pool_size;
					}
				}
			}
		}
		bottom_diff[index] = gradient;
	}
}

template<typename Dtype>
void ShiftPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (!propagate_down[0]) {
		return;
	}
	const Dtype* top_diff = top[0]->gpu_diff();
	Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
	const int count = bottom[0]->count();
	caffe_gpu_set(count, Dtype(0.), bottom_diff);
	// We'll output the mask to top[1] if it's of size >1.
	const bool use_top_mask = top.size() > 1;
	const int* mask = NULL;
	const Dtype* top_mask = NULL;
	switch (this->layer_param_.pooling_param().pool()) {
	case PoolingParameter_PoolMethod_MAX:
		// The main loop
		if (use_top_mask) {
			top_mask = top[1]->cpu_data();
		} else {
			mask = max_idx_.cpu_data();
		}
		ShiftMaxPoolBackward<Dtype> <<<CAFFE_GET_BLOCKS(count),
				CAFFE_CUDA_NUM_THREADS>>>(count, top_diff, mask, top_mask,
				bottom[0]->num(), channels_, height_, width_, pooled_height_,
				pooled_width_, kernel_h_, kernel_w_, stride_h_, stride_w_, pad_h_,
				pad_w_, bottom_diff);
		break;
	case PoolingParameter_PoolMethod_AVE:
		ShiftAvePoolBackward<Dtype> <<<CAFFE_GET_BLOCKS(count),
				CAFFE_CUDA_NUM_THREADS>>>(count, top_diff, bottom[0]->num(), channels_,
				height_, width_, pooled_height_, pooled_width_, kernel_h_, kernel_w_,
				stride_h_, stride_w_, pad_h_, pad_w_, bottom_diff);
		break;
	case PoolingParameter_PoolMethod_STOCHASTIC:
		NOT_IMPLEMENTED;
		break;
	default:
		LOG(FATAL)<< "Unknown pooling method.";
	}

	CUDA_POST_KERNEL_CHECK
	;
}

INSTANTIATE_LAYER_GPU_FUNCS(ShiftPoolingLayer);

} // namespace caffe
