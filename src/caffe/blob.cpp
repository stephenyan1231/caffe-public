#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/net.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
void Blob<Dtype>::Reshape(const int num, const int channels, const int height,
		const int width) {
	CHECK_GE(num, 0);
	CHECK_GE(channels, 0);
	CHECK_GE(height, 0);
	CHECK_GE(width, 0);
	num_ = num;
	channels_ = channels;
	height_ = height;
	width_ = width;
	count_ = num_ * channels_ * height_ * width_;
	if (count_ > capacity_) {
//		LOG(INFO)<<"Blob<Dtype>::Reshape capacity "<<capacity_<<"--->"<<count_<<" "
//				<<count_*sizeof(Dtype);
		capacity_ = count_;
		data_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
		diff_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
	}
}

template<typename Dtype>
void Blob<Dtype>::ReshapeLike(const Blob<Dtype>& other) {
	Reshape(other.num(), other.channels(), other.height(), other.width());
}

template<typename Dtype>
void Blob<Dtype>::ReshapeForceMemoryFree(const int num, const int channels, const int height,
		const int width) {
	CHECK_GE(num, 0);
	CHECK_GE(channels, 0);
	CHECK_GE(height, 0);
	CHECK_GE(width, 0);

	num_ = num;
	channels_ = channels;
	height_ = height;
	width_ = width;
	count_ = num_ * channels_ * height_ * width_;
	if (count_ != capacity_) {
//		LOG(INFO)<<"Blob<Dtype>::ReshapeForceMemoryFree capacity_ "<<capacity_<<
//				"--->"<<count_;
//		this->gpu_data();
//		LOG(INFO)<<"Blob<Dtype>::ReshapeForceMemoryFree "<<capacity_<<"---->"<<count_
//				<<" "<<count_*sizeof(Dtype);
		capacity_ = count_;
		data_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
		diff_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
//		if(capacity_ > 0){
//			data_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
//			diff_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
//		} else {
//			LOG(INFO)<<"release managed SyncedMemory";
//			data_.reset();
//			diff_.reset();
//		}
	}
}

// make sure only the gpu data of the reshaped blob will be used
// avoid data copy from device to host by omitting set_cpu_data and set_cpu_diff
template<typename Dtype>
shared_ptr<Blob<Dtype> > Blob<Dtype>::ReshapedGPUOnly(const int num, const int channels, const int height,
  const int width){
	CHECK_EQ(count_, num * channels * height * width);
//	LOG(INFO)<<"Blob<Dtype>::ReshapedGPUOnly count "<<count_<<" "
//			<<count_*sizeof(Dtype);
	shared_ptr<Blob<Dtype> > new_blob = shared_ptr<Blob<Dtype> >
	(new Blob<Dtype>(num, channels, height, width));
	new_blob->set_gpu_data(mutable_gpu_data(), Caffe::GetDeviceId());
	new_blob->set_gpu_diff(mutable_gpu_diff(), Caffe::GetDeviceId());
	return new_blob;
}


template<typename Dtype>
Blob<Dtype>::Blob(const int num, const int channels, const int height,
		const int width):
// capacity_ must be initialized before calling Reshape
		capacity_(0){
	Reshape(num, channels, height, width);
}

template<typename Dtype>
const Dtype* Blob<Dtype>::cpu_data() const {
	CHECK(data_);
	return (const Dtype*) data_->cpu_data();
}

template<typename Dtype>
void Blob<Dtype>::set_cpu_data(Dtype* data) {
	CHECK(data);
	data_->set_cpu_data(data);
}

template<typename Dtype>
const Dtype* Blob<Dtype>::gpu_data() const {
	CHECK(data_);
	return (const Dtype*) data_->gpu_data();
}

template<typename Dtype>
void Blob<Dtype>::set_gpu_data(Dtype* data, int device_id) {
	CHECK(data);
	data_->set_gpu_data(data, device_id);
}

template<typename Dtype>
const Dtype* Blob<Dtype>::cpu_diff() const {
	CHECK(diff_);
	return (const Dtype*) diff_->cpu_data();
}

template<typename Dtype>
void Blob<Dtype>::set_cpu_diff(Dtype* diff) {
	CHECK(diff);
	diff_->set_cpu_data(diff);
}

template<typename Dtype>
const Dtype* Blob<Dtype>::gpu_diff() const {
	CHECK(diff_);
	return (const Dtype*) diff_->gpu_data();
}

template<typename Dtype>
void Blob<Dtype>::set_gpu_diff(Dtype* diff, int device_id) {
	CHECK(diff);
	diff_->set_gpu_data(diff, device_id);
}

template<typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_data() {
	CHECK(data_);
	return static_cast<Dtype*>(data_->mutable_cpu_data());
}

template<typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_data() {
	CHECK(data_);
	return static_cast<Dtype*>(data_->mutable_gpu_data());
}

template<typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_diff() {
	CHECK(diff_);
	return static_cast<Dtype*>(diff_->mutable_cpu_data());
}

template<typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_diff() {
	CHECK(diff_);
	return static_cast<Dtype*>(diff_->mutable_gpu_data());
}

template<typename Dtype>
void Blob<Dtype>::ShareData(const Blob& other) {
	CHECK_EQ(count_, other.count());
	data_ = other.data();
}

template<typename Dtype>
void Blob<Dtype>::ShareDiff(const Blob& other) {
	CHECK_EQ(count_, other.count());
	diff_ = other.diff();
}

// make sure only the gpu data of the returned blob will be used
// avoid data copy from device to host by omitting set_cpu_data and set_cpu_diff
template<typename Dtype>
shared_ptr<Blob<Dtype> > Blob<Dtype>::SliceNumGPUOnly(int start_num, int end_num) {
	CHECK_GT(end_num - start_num, 0);
	shared_ptr<Blob<Dtype> > new_blob = shared_ptr<Blob<Dtype> >
	(new Blob<Dtype>(end_num - start_num, channels_,
			height_, width_));

//	new_blob->set_cpu_data(mutable_cpu_data() + offset(start_num));
//	new_blob->set_cpu_diff(mutable_cpu_diff() + offset(start_num));
	new_blob->set_gpu_data(mutable_gpu_data() + offset(start_num),
			Caffe::GetDeviceId());
	new_blob->set_gpu_diff(mutable_gpu_diff() + offset(start_num),
			Caffe::GetDeviceId());

	return new_blob;
}

// The "update" method is used for parameter blobs in a Net, which are stored
// as Blob<float> or Blob<double> -- hence we do not define it for
// Blob<int> or Blob<unsigned int>.
template<> void Blob<unsigned int>::Update() {
	NOT_IMPLEMENTED;
}
template<> void Blob<int>::Update() {
	NOT_IMPLEMENTED;
}
template<> void Blob<unsigned short>::Update() {
	NOT_IMPLEMENTED;
}

template<typename Dtype>
void Blob<Dtype>::Update() {
	// We will perform update based on where the data is located.
	switch (data_->head()) {
	case SyncedMemory::HEAD_AT_CPU:
		// perform computation on CPU
		caffe_axpy<Dtype>(count_, Dtype(-1),
				static_cast<const Dtype*>(diff_->cpu_data()),
				static_cast<Dtype*>(data_->mutable_cpu_data()));
		break;
	case SyncedMemory::HEAD_AT_GPU:
	case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
		// perform computation on GPU
		caffe_gpu_axpy<Dtype>(count_, Dtype(-1),
				static_cast<const Dtype*>(diff_->gpu_data()),
				static_cast<Dtype*>(data_->mutable_gpu_data()));
#else
		NO_GPU;
#endif
		break;
	default:
		LOG(FATAL)<< "Syncedmem not initialized.";
	}
}

template<> unsigned int Blob<unsigned int>::asum_data() const {
	NOT_IMPLEMENTED;
	return 0;
}

template<> int Blob<int>::asum_data() const {
	NOT_IMPLEMENTED;
	return 0;
}

template<> unsigned short Blob<unsigned short>::asum_data() const {
	NOT_IMPLEMENTED;
	return 0;
}


template<typename Dtype>
Dtype Blob<Dtype>::asum_data() const {
	if (!data_) {
		return 0;
	}
	switch (data_->head()) {
	case SyncedMemory::HEAD_AT_CPU:
		return caffe_cpu_asum(count_, cpu_data());
	case SyncedMemory::HEAD_AT_GPU:
	case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
	{
		Dtype asum;
		caffe_gpu_asum(count_, gpu_data(), &asum);
		return asum;
	}
#else
		NO_GPU;
#endif
	case SyncedMemory::UNINITIALIZED:
		return 0;
	default:
		LOG(FATAL)<< "Unknown SyncedMemory head state: " << data_->head();
	}
	return 0;
}

template<> unsigned int Blob<unsigned int>::asum_diff() const {
	NOT_IMPLEMENTED;
	return 0;
}

template<> int Blob<int>::asum_diff() const {
	NOT_IMPLEMENTED;
	return 0;
}

template<> unsigned short Blob<unsigned short>::asum_diff() const {
	NOT_IMPLEMENTED;
	return 0;
}

template<typename Dtype>
Dtype Blob<Dtype>::asum_diff() const {
	if (!diff_) {
		return 0;
	}
	switch (diff_->head()) {
	case SyncedMemory::HEAD_AT_CPU:
		return caffe_cpu_asum(count_, cpu_diff());
	case SyncedMemory::HEAD_AT_GPU:
	case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
	{
		Dtype asum;
		caffe_gpu_asum(count_, gpu_diff(), &asum);
		return asum;
	}
#else
		NO_GPU;
#endif
	case SyncedMemory::UNINITIALIZED:
		return 0;
	default:
		LOG(FATAL)<< "Unknown SyncedMemory head state: " << diff_->head();
	}
	return 0;
}

template<> unsigned int Blob<unsigned int>::sumsq_data() const {
	NOT_IMPLEMENTED;
	return 0;
}

template<> int Blob<int>::sumsq_data() const {
	NOT_IMPLEMENTED;
	return 0;
}

template<> unsigned short Blob<unsigned short>::sumsq_data() const {
	NOT_IMPLEMENTED;
	return 0;
}

template<typename Dtype>
Dtype Blob<Dtype>::sumsq_data() const {
	Dtype sumsq;
	const Dtype* data;
	if (!data_) {
		return 0;
	}
	switch (data_->head()) {
	case SyncedMemory::HEAD_AT_CPU:
		data = cpu_data();
		sumsq = caffe_cpu_dot(count_, data, data);
		break;
	case SyncedMemory::HEAD_AT_GPU:
	case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
		data = gpu_data();
		caffe_gpu_dot(count_, data, data, &sumsq);
#else
		NO_GPU;
#endif
		break;
	case SyncedMemory::UNINITIALIZED:
		return 0;
	default:
		LOG(FATAL)<< "Unknown SyncedMemory head state: " << data_->head();
	}
	return sumsq;
}

template<> unsigned int Blob<unsigned int>::sumsq_diff() const {
	NOT_IMPLEMENTED;
	return 0;
}

template<> int Blob<int>::sumsq_diff() const {
	NOT_IMPLEMENTED;
	return 0;
}

template<> unsigned short Blob<unsigned short>::sumsq_diff() const {
	NOT_IMPLEMENTED;
	return 0;
}

template<typename Dtype>
Dtype Blob<Dtype>::sumsq_diff() const {
	Dtype sumsq;
	const Dtype* diff;
	if (!diff_) {
		return 0;
	}
	switch (diff_->head()) {
	case SyncedMemory::HEAD_AT_CPU:
		diff = cpu_diff();
		sumsq = caffe_cpu_dot(count_, diff, diff);
		break;
	case SyncedMemory::HEAD_AT_GPU:
	case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
		diff = gpu_diff();
		caffe_gpu_dot(count_, diff, diff, &sumsq);
		break;
#else
		NO_GPU;
#endif
	case SyncedMemory::UNINITIALIZED:
		return 0;
	default:
		LOG(FATAL)<< "Unknown SyncedMemory head state: " << data_->head();
	}
	return sumsq;
}

template<typename Dtype>
void Blob<Dtype>::CopyFrom(const Blob& source, bool copy_diff, bool reshape) {
	if (num_ != source.num() || channels_ != source.channels()
			|| height_ != source.height() || width_ != source.width()) {
		if (reshape) {
			Reshape(source.num(), source.channels(), source.height(), source.width());
		} else {
			LOG(FATAL)<< "Trying to copy blobs of different sizes.";
		}
	}
	switch (Caffe::mode()) {
		case Caffe::GPU:
		if (copy_diff) {
			caffe_copy(count_, source.gpu_diff(),
					static_cast<Dtype*>(diff_->mutable_gpu_data()));
		} else {
			caffe_copy(count_, source.gpu_data(),
					static_cast<Dtype*>(data_->mutable_gpu_data()));
		}
		break;
		case Caffe::CPU:
		if (copy_diff) {
			caffe_copy(count_, source.cpu_diff(),
					static_cast<Dtype*>(diff_->mutable_cpu_data()));
		} else {
			caffe_copy(count_, source.cpu_data(),
					static_cast<Dtype*>(data_->mutable_cpu_data()));
		}
		break;
		default:
		LOG(FATAL) << "Unknown caffe mode.";
	}
}

template<typename Dtype>
void Blob<Dtype>::CopyFrom(const vector<Blob<Dtype>*> &sources, bool copy_diff,
		bool reshape) {
	if (sources.size() == 0) {
		LOG(FATAL)<< "Trying to copy from empty sources";
	}
	int num_sum = 0;
	int channels = sources[0]->channels();
	int height = sources[0]->height();
	int width = sources[0]->width();

	for(int i = 0;i < sources.size();++i) {
		if(channels != sources[i]->channels() || height != sources[i]->height() || width != sources[i]->width()) {
			LOG(FATAL) << "Trying to combine blobs of different sizes.";
		}
		num_sum += sources[i]->num();
	}
	if(!reshape) {
		if(num_ != num_sum || channels_ != channels || height_ != height || width_ != width) {
			LOG(FATAL) << "Trying to copy from empty sources";
		}
	}
	else {
		Reshape(num_sum, channels, height, width);
	}

	if(copy_diff) {
		Dtype* diff_ptr;
		switch (Caffe::mode()) {
			case Caffe::GPU:
			diff_ptr = static_cast<Dtype*>(diff_->mutable_gpu_data());
			for(int i=0;i<sources.size();++i) {
				caffe_copy(sources[i]->count(), sources[i]->gpu_diff(), diff_ptr);
				diff_ptr += sources[i]->offset(sources[i]->num());
			}
			break;
			case Caffe::CPU:
			diff_ptr = static_cast<Dtype*> (diff_->mutable_cpu_data());
			for(int i=0;i<sources.size();++i) {
				caffe_copy(sources[i]->count(), sources[i]->cpu_diff(), diff_ptr);
				diff_ptr += sources[i]->offset(sources[i]->num());
			}
			break;
			default:
			LOG(FATAL) << "Unknown caffe mode.";
		}
	}
	else {
		Dtype* data_ptr;
		switch (Caffe::mode()) {
			case Caffe::GPU:
			data_ptr = static_cast<Dtype*>(data_->mutable_gpu_data());
			for(int i=0;i<sources.size();++i) {
				caffe_copy(sources[i]->count(), sources[i]->gpu_data(), data_ptr);
				data_ptr += sources[i]->offset(sources[i]->num());
			}
			break;
			case Caffe::CPU:
			data_ptr = static_cast<Dtype*>(diff_->mutable_cpu_data());
			for(int i=0;i<sources.size();++i) {
				caffe_copy(sources[i]->count(), sources[i]->cpu_data(), data_ptr);
				data_ptr += sources[i]->offset(sources[i]->num());
			}
			break;
			default:
			LOG(FATAL) << "Unknown caffe mode.";
		}
	}

}

template<typename Dtype>
void Blob<Dtype>::FromProto(const BlobProto& proto) {
	Reshape(proto.num(), proto.channels(), proto.height(), proto.width());
	// copy data
	Dtype* data_vec = mutable_cpu_data();
	for (int i = 0; i < count_; ++i) {
		data_vec[i] = proto.data(i);
	}
	if (proto.diff_size() > 0) {
		Dtype* diff_vec = mutable_cpu_diff();
		for (int i = 0; i < count_; ++i) {
			diff_vec[i] = proto.diff(i);
		}
	}
}

template<typename Dtype>
void Blob<Dtype>::ToProto(BlobProto* proto, bool write_diff) const {
	proto->set_num(num_);
	proto->set_channels(channels_);
	proto->set_height(height_);
	proto->set_width(width_);
	proto->clear_data();
	proto->clear_diff();
	const Dtype* data_vec = cpu_data();
	for (int i = 0; i < count_; ++i) {
		proto->add_data(data_vec[i]);
	}
	if (write_diff) {
		const Dtype* diff_vec = cpu_diff();
		for (int i = 0; i < count_; ++i) {
			proto->add_diff(diff_vec[i]);
		}
	}
}

INSTANTIATE_CLASS(Blob);
template class Blob<int> ;
template class Blob<unsigned int> ;
template class Blob<unsigned short>;

}  // namespace caffe

