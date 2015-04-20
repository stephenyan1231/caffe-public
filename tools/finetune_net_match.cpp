// Copyright 2014 Zhicheng Yan@eBay
// Aug 7, 2014
// similar to finetune_net.cpp except that
// For an already initialized net, CopyTrainedLayersFromPrefixMatching() copies the already
// trained layers from other net parameter instances as long as the source layer name is a prefix
// of a target layer name
#include <glog/logging.h>
#include <cuda_runtime.h>

#include <string>

#include "caffe/caffe.hpp"

using namespace caffe;
// NOLINT(build/namespaces)

DEFINE_string(match_mode, "EXACT_MATCH", "the matching mode used to copy layer parameter from pretrained model");
DEFINE_string(solver, "", "The solver definition protocol buffer text file.");

int main(int argc, char** argv) {
	caffe::GlobalInit(&argc, &argv);
	if (argc < 2) {
		LOG(ERROR)
				<< "Usage: finetune_net_match [args] pretrained_net_1 pretrained_net_2 ...";
		return 1;
	}

	SolverParameter solver_param;
	ReadProtoFromTextFileOrDie(FLAGS_solver.c_str(), &solver_param);

	if(solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU){
		Caffe::set_mode(Caffe::GPU);
	}

	LOG(INFO) << "Starting Optimization";
	shared_ptr<caffe::Solver<float> >
	solver(caffe::GetSolver<float>(solver_param));
	std::vector < string > params;
	for (int i = 1; i < argc; ++i) {
		LOG(INFO) << "Loading from " << argv[i];
		params.push_back(string(argv[i]));
	}

	if(FLAGS_match_mode == std::string("EXACT_MATCH")){
		LOG(INFO)<<"exact matching mode";
		solver->net()->CopyTrainedLayersFrom(params);
	}else if(FLAGS_match_mode == std::string("SUFFIX_MATCH")){
		LOG(INFO)<<"suffix matching mode";
		solver->net()->CopyTrainedLayersFromSuffixMatch(params);
	}else{
		LOG(FATAL)<<"unknow matching mode: "<<FLAGS_match_mode;
	}
	solver->Solve();
	LOG(INFO) << "Optimization Done.";

	return 0;
}
