// Zhicheng Yan@eBay
// Aug 7, 2014
// similar to finetune_net.cpp except that
// For an already initialized net, CopyTrainedLayersFromPrefixMatching() copies the already
// trained layers from other net parameter instances as long as the source layer name is a prefix
// of a target layer name

#include <cuda_runtime.h>

#include <string>

#include "caffe/caffe.hpp"

using namespace caffe;
// NOLINT(build/namespaces)

int main(int argc, char** argv) {
	::google::InitGoogleLogging(argv[0]);
	if (argc < 3) {
		LOG(ERROR)
				<< "Usage: finetune_net_exact_match solver_proto_file pretrained_net_1 pretrained_net_2 ...";
		return 1;
	}

	SolverParameter solver_param;
	ReadProtoFromTextFileOrDie(argv[1], &solver_param);

	LOG(INFO) << "Starting Optimization";
	SGDSolver<float> solver(solver_param);
	std::vector < string > params;
	for (int i = 2; i < argc; ++i) {
		LOG(INFO) << "Loading from " << argv[i];
		params.push_back(string(argv[i]));
	}

	solver.net()->CopyTrainedLayersFrom(params);
	solver.Solve();
	LOG(INFO) << "Optimization Done.";

	return 0;
}
