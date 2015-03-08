#include <glog/logging.h>
#include <cstdio>
#include <ctime>

#include "caffe/common.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

shared_ptr<Caffe> Caffe::singleton_;

//set<int> Caffe::device_ids_;
//// device_id -> cublasHandle
//map<int, cublasHandle_t> Caffe::cublas_handle_;
//map<int, curandGenerator_t> Caffe::Get().curand_generator_;

//std::map<int, cudaStream_t> Caffe::Get().default_streams_;
//boost::mutex Caffe::Get().stream_mutex_;
//boost::mutex Caffe::Get().cublas_mutex_;
//boost::mutex Caffe::Get().curand_mutex_;

// random seeding
int64_t cluster_seedgen(void) {
	int64_t s, seed, pid;
	FILE* f = fopen("/dev/urandom", "rb");
	if (f && fread(&seed, 1, sizeof(seed), f) == sizeof(seed)) {
		fclose(f);
		return seed;
	}

	LOG(INFO)<< "System entropy source not available, "
	"using fallback algorithm to generate seed instead.";
	if (f)
		fclose(f);

	pid = getpid();
	s = time(NULL);
	seed = abs(((s * 181) * ((pid - 83) * 359)) % 104729);
	return seed;
}

void GlobalInit(int* pargc, char*** pargv) {
	// Google flags.
	::gflags::ParseCommandLineFlags(pargc, pargv, true);
	// Google logging.
	::google::InitGoogleLogging(*(pargv)[0]);
	// Provide a backtrace on segfault.
	::google::InstallFailureSignalHandler();
}

#ifdef CPU_ONLY  // CPU-only Caffe.
Caffe::Caffe()
: random_generator_(), mode_(Caffe::CPU), phase_(Caffe::TRAIN) {}

Caffe::~Caffe() {}

void Caffe::set_random_seed(const unsigned int seed) {
	// RNG seed
	Get().random_generator_.reset(new RNG(seed));
}

void Caffe::SetDevice(const int device_id) {
	NO_GPU;
}

void Caffe::DeviceQuery() {
	NO_GPU;
}

class Caffe::RNG::Generator {
public:
	Generator() : rng_(new caffe::rng_t(cluster_seedgen())) {}
	explicit Generator(unsigned int seed) : rng_(new caffe::rng_t(seed)) {}
	caffe::rng_t* rng() {return rng_.get();}
private:
	shared_ptr<caffe::rng_t> rng_;
};

Caffe::RNG::RNG() : generator_(new Generator()) {}

Caffe::RNG::RNG(unsigned int seed) : generator_(new Generator(seed)) {}

Caffe::RNG& Caffe::RNG::operator=(const RNG& other) {
	generator_ = other.generator_;
	return *this;
}

void* Caffe::RNG::generator() {
	return static_cast<void*>(generator_->rng());
}

#else  // Normal GPU + CPU Caffe.
Caffe::Caffe() :
//		cublas_handle_(NULL), Get().curand_generator_(NULL),
		random_generator_(), mode_(Caffe::CPU), phase_(Caffe::TRAIN) {
	// Try to create a cublas handler, and report an error if failed (but we will
	// keep the program running as one might just want to run CPU code).
//	if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS) {
//		LOG(ERROR)<< "Cannot create Cublas handle. Cublas won't be available.";
//	}
	// Try to create a curand handler.
//	if (curandCreateGenerator(&Get().curand_generator_, CURAND_RNG_PSEUDO_DEFAULT)
//			!= CURAND_STATUS_SUCCESS ||
//			curandSetPseudoRandomGeneratorSeed(Get().curand_generator_, cluster_seedgen())
//			!= CURAND_STATUS_SUCCESS) {
//		LOG(ERROR) << "Cannot create Curand generator. Curand won't be available.";
//	}
}

Caffe::~Caffe() {
//	if (cublas_handle_)
//		CUBLAS_CHECK(cublasDestroy(cublas_handle_));
//	if (Get().curand_generator_) {
//		CURAND_CHECK(curandDestroyGenerator(Get().curand_generator_));
//	}
	for (std::map<int, cublasHandle_t>::iterator it =
			Get().cublas_handle_.begin(); it != Get().cublas_handle_.end(); ++it) {
		CUBLAS_CHECK(cublasDestroy(it->second));
	}
	for (std::map<int, curandGenerator_t>::iterator it =
			Get().curand_generator_.begin(); it != Get().curand_generator_.end();
			++it) {
		CURAND_CHECK(curandDestroyGenerator(it->second));
	}
}

void Caffe::set_random_seed(const unsigned int seed) {
	// Curand seed
	static bool g_curand_availability_logged = false;
	if (Get().curand_generator_.size() > 0) {
		for (std::map<int, curandGenerator_t>::iterator it =
				Get().curand_generator_.begin(); it != Get().curand_generator_.end();
				it++) {
			CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(it->second, seed));
			CURAND_CHECK(curandSetGeneratorOffset(it->second, 0));
		}

	} else {
		if (!g_curand_availability_logged) {
			LOG(ERROR)<<
			"Curand not available. Skipping setting the curand seed.";
			g_curand_availability_logged = true;
		}
	}
	// RNG seed
	Get().random_generator_.reset(new RNG(seed));
}

void Caffe::SetDevice(const int device_id) {
	CUDA_CHECK(cudaSetDevice(device_id));
//  int current_device;
//  CUDA_CHECK(cudaGetDevice(&current_device));
//  if (current_device == device_id) {
//    return;
//  }
//  // The call to cudaSetDevice must come before any calls to Get, which
//  // may perform initialization using the GPU.
//  CUDA_CHECK(cudaSetDevice(device_id));
//  if (Get().cublas_handle_) CUBLAS_CHECK(cublasDestroy(Get().cublas_handle_));
//  if (Get()..curand_generator_) {
//    CURAND_CHECK(curandDestroyGenerator(Get()..curand_generator_));
//  }
//  CUBLAS_CHECK(cublasCreate(&Get().cublas_handle_));
//  CURAND_CHECK(curandCreateGenerator(&Get()..curand_generator_,
//      CURAND_RNG_PSEUDO_DEFAULT));
//  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(Get()..curand_generator_,
//      cluster_seedgen()));
}

void Caffe::DeviceQuery() {
	cudaDeviceProp prop;
	int device;
	if (cudaSuccess != cudaGetDevice(&device)) {
		printf("No cuda device present.\n");
		return;
	}
	CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
	LOG(INFO)<< "Device id:                     " << device;
	LOG(INFO)<< "Major revision number:         " << prop.major;
	LOG(INFO)<< "Minor revision number:         " << prop.minor;
	LOG(INFO)<< "Name:                          " << prop.name;
	LOG(INFO)<< "Total global memory:           " << prop.totalGlobalMem;
	LOG(INFO)<< "Total shared memory per block: " << prop.sharedMemPerBlock;
	LOG(INFO)<< "Total registers per block:     " << prop.regsPerBlock;
	LOG(INFO)<< "Warp size:                     " << prop.warpSize;
	LOG(INFO)<< "Maximum memory pitch:          " << prop.memPitch;
	LOG(INFO)<< "Maximum threads per block:     " << prop.maxThreadsPerBlock;
	LOG(INFO)<< "Maximum dimension of block:    "
	<< prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", "
	<< prop.maxThreadsDim[2];
	LOG(INFO)<< "Maximum dimension of grid:     "
	<< prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", "
	<< prop.maxGridSize[2];
	LOG(INFO)<< "Clock rate:                    " << prop.clockRate;
	LOG(INFO)<< "Total constant memory:         " << prop.totalConstMem;
	LOG(INFO)<< "Texture alignment:             " << prop.textureAlignment;
	LOG(INFO)<< "Concurrent copy and execution: "
	<< (prop.deviceOverlap ? "Yes" : "No");
	LOG(INFO)<< "Number of multiprocessors:     " << prop.multiProcessorCount;
	LOG(INFO)<< "Kernel execution timeout:      "
	<< (prop.kernelExecTimeoutEnabled ? "Yes" : "No");
	return;
}

int Caffe::GetDeviceId() {
	int d = 0;
	if(Caffe::mode() == GPU){
		CUDA_CHECK(cudaGetDevice(&d));
	}
	return d;
}

cublasHandle_t Caffe::cublas_handle(int device_id) {
	Get().cublas_mutex_.lock();
	CHECK_EQ(Get().cublas_handle_.count(device_id),1);
	cublasHandle_t ret = Get().cublas_handle_[device_id];
	Get().cublas_mutex_.unlock();
	return ret;
}

void Caffe::InitDevices(const std::vector<int> &device_id) {
	for (int i = 0; i < device_id.size(); ++i) {
		InitDevice(device_id[i]);
	}
}

void Caffe::InitDevice(int device_id) {
	if (Get().device_ids_.count(device_id) != 0) {
		return;
	}
	Get().device_ids_.insert(device_id);

	// The call to cudaSetDevice must come before any calls to Get, which
	// may perform initialization using the GPU.
//  CUDA_CHECK(cudaSetDevice(device_id));
//  if (Get().cublas_handle_) CUBLAS_CHECK(cublasDestroy(Get().cublas_handle_));
//  if (Get()..curand_generator_) {
//    CURAND_CHECK(curandDestroyGenerator(Get()..curand_generator_));
//  }
	SetDevice(device_id);

	LOG(INFO)<<"Caffe::InitDevice(int device_id) p1";
	cublasHandle_t cublas_handle;
	CUBLAS_CHECK(cublasCreate(&cublas_handle));
	Get().cublas_handle_[device_id] = cublas_handle;

	LOG(INFO)<<"Caffe::InitDevice(int device_id) p2";

	curandGenerator_t curand_generator;
	CURAND_CHECK(
			curandCreateGenerator(&curand_generator, CURAND_RNG_PSEUDO_DEFAULT));
	Get().curand_generator_[device_id] = curand_generator;

	CURAND_CHECK(
			curandSetPseudoRandomGeneratorSeed(Get().curand_generator_[device_id],
					cluster_seedgen()));
	LOG(INFO)<<"Caffe::InitDevice(int device_id) p3";

	SyncDevice();
}

void Caffe::SyncDevice() {
	CUDA_CHECK(cudaDeviceSynchronize());
}

//void Caffe::SyncStream(){
//	SyncStream(Caffe::GetDefaultStream());
//}

void Caffe::SyncStream(cudaStream_t stream){
	CUDA_CHECK(cudaStreamSynchronize(stream));
}

void Caffe::CublasSetStream(cublasHandle_t handle){
	CUBLAS_CHECK(cublasSetStream(handle, 0));
}

void Caffe::CublasSetStream(cublasHandle_t handle, cudaStream_t stream){
	CUBLAS_CHECK(cublasSetStream(handle, stream));
}


bool Caffe::CanAccessPeer(int src_device, int tgt_device){
	if(src_device == tgt_device){
		return true;
	}
	int can_access;
	CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, src_device, tgt_device));
	return can_access;
}

//// get default non-null stream on the current device
//cudaStream_t Caffe::GetDefaultStream() {
//	return GetDefaultStream(GetDeviceId());
//}
//
//cudaStream_t Caffe::GetDefaultStream(int device_id) {
//	if (device_id >= 0) {
////		Get().stream_mutex_.lock();
//		Get().default_stream_mutex_.lock();
//		if (Get().default_streams_.count(device_id) == 0) {
//			int old_device = GetDeviceId();
//			SetDevice(device_id);
//			CUDA_CHECK(
//					cudaStreamCreateWithFlags(&Get().default_streams_[device_id],
//							cudaStreamNonBlocking));
//			SetDevice(old_device);
//		}
//		cudaStream_t s = Get().default_streams_[device_id];
//		Get().default_stream_mutex_.unlock();
////		Get().stream_mutex_.unlock();
//		return s;
//	}
//	return 0;
//}

class Caffe::RNG::Generator {
public:
	Generator() :
			rng_(new caffe::rng_t(cluster_seedgen())) {
	}
	explicit Generator(unsigned int seed) :
			rng_(new caffe::rng_t(seed)) {
	}
	caffe::rng_t* rng() {
		return rng_.get();
	}
private:
	shared_ptr<caffe::rng_t> rng_;
};

Caffe::RNG::RNG() :
		generator_(new Generator()) {
}

Caffe::RNG::RNG(unsigned int seed) :
		generator_(new Generator(seed)) {
}

Caffe::RNG& Caffe::RNG::operator=(const RNG& other) {
	generator_.reset(other.generator_.get());
	return *this;
}

void* Caffe::RNG::generator() {
	return static_cast<void*>(generator_->rng());
}

const char* cublasGetErrorString(cublasStatus_t error) {
	switch (error) {
	case CUBLAS_STATUS_SUCCESS:
		return "CUBLAS_STATUS_SUCCESS";
	case CUBLAS_STATUS_NOT_INITIALIZED:
		return "CUBLAS_STATUS_NOT_INITIALIZED";
	case CUBLAS_STATUS_ALLOC_FAILED:
		return "CUBLAS_STATUS_ALLOC_FAILED";
	case CUBLAS_STATUS_INVALID_VALUE:
		return "CUBLAS_STATUS_INVALID_VALUE";
	case CUBLAS_STATUS_ARCH_MISMATCH:
		return "CUBLAS_STATUS_ARCH_MISMATCH";
	case CUBLAS_STATUS_MAPPING_ERROR:
		return "CUBLAS_STATUS_MAPPING_ERROR";
	case CUBLAS_STATUS_EXECUTION_FAILED:
		return "CUBLAS_STATUS_EXECUTION_FAILED";
	case CUBLAS_STATUS_INTERNAL_ERROR:
		return "CUBLAS_STATUS_INTERNAL_ERROR";
#if CUDA_VERSION >= 6000
	case CUBLAS_STATUS_NOT_SUPPORTED:
		return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
#if CUDA_VERSION >= 6050
	case CUBLAS_STATUS_LICENSE_ERROR:
		return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
	}
	return "Unknown cublas status";
}

const char* curandGetErrorString(curandStatus_t error) {
	switch (error) {
	case CURAND_STATUS_SUCCESS:
		return "CURAND_STATUS_SUCCESS";
	case CURAND_STATUS_VERSION_MISMATCH:
		return "CURAND_STATUS_VERSION_MISMATCH";
	case CURAND_STATUS_NOT_INITIALIZED:
		return "CURAND_STATUS_NOT_INITIALIZED";
	case CURAND_STATUS_ALLOCATION_FAILED:
		return "CURAND_STATUS_ALLOCATION_FAILED";
	case CURAND_STATUS_TYPE_ERROR:
		return "CURAND_STATUS_TYPE_ERROR";
	case CURAND_STATUS_OUT_OF_RANGE:
		return "CURAND_STATUS_OUT_OF_RANGE";
	case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
		return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
	case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
		return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
	case CURAND_STATUS_LAUNCH_FAILURE:
		return "CURAND_STATUS_LAUNCH_FAILURE";
	case CURAND_STATUS_PREEXISTING_FAILURE:
		return "CURAND_STATUS_PREEXISTING_FAILURE";
	case CURAND_STATUS_INITIALIZATION_FAILED:
		return "CURAND_STATUS_INITIALIZATION_FAILED";
	case CURAND_STATUS_ARCH_MISMATCH:
		return "CURAND_STATUS_ARCH_MISMATCH";
	case CURAND_STATUS_INTERNAL_ERROR:
		return "CURAND_STATUS_INTERNAL_ERROR";
	}
	return "Unknown curand status";
}

#endif  // CPU_ONLY
std::vector<int> parse_int_list(std::string s, std::string delimiter) {
	std::vector<int> intv;
	if (s.length() == 0) {
		return intv;
	}
	size_t pos = 0;
	std::string token;
	int id;
	while ((pos = s.find(delimiter)) != std::string::npos) {
		token = s.substr(0, pos);
		if (sscanf(token.c_str(), "%d", &id) < 1) {
			LOG(ERROR)<< "can not parse int list "<<token;
		}
		else {
			intv.push_back(id);
			s.erase(0,pos+delimiter.length());
		}
	}
	if (sscanf(s.c_str(), "%d", &id) < 1) {
		LOG(ERROR)<< "can not parse int list "<<s;
	}
	else {
		intv.push_back(id);
	}
	return intv;
}

int divide_up(int n, int m){
	return (n + m -1) / m;
}

}  // namespace caffe
