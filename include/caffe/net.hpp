#ifndef CAFFE_NET_HPP_
#define CAFFE_NET_HPP_

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/data_manager.hpp"
#include "caffe/work_message.hpp"
//#include "caffe/net_thread_solver.hpp"
//#include "caffe/blob_diff_reducer.hpp"
#include "caffe/copy_pipeline.hpp"

namespace caffe {

template<typename Dtype>
class NetThread;

template <typename Dtype>
class Solver;

template <typename Dtype>
class BlobSolver;
/**
 * @brief Connects Layer%s together into a directed acyclic graph (DAG)
 *        specified by a NetParameter.
 *
 * TODO(dox): more thorough description.
 */
template <typename Dtype>
class Net {
 public:
  explicit Net(const NetParameter& param, std::vector<int> device_ids = std::vector<int>(),
  		Solver<Dtype> *solver = NULL, SolverParameter solver_param = SolverParameter());
  explicit Net(const string& param_file, std::vector<int> device_ids = std::vector<int>(),
  		Solver<Dtype> *solver = NULL, SolverParameter solver_param = SolverParameter());
  virtual ~Net();

  void Init(const NetParameter& in_param, vector<int> &device_ids);
  void PostInit();

  void InitDataManager(const NetParameter& param);
  void InitNetThreads(const NetParameter& param);
  void ConnectReplicas();
  vector<int>& GetDeviceIds(){return device_ids_;}

  void SetBatchSize(int replica_id, int batch_size){
  	CHECK_GT(batch_sizes_.size(), replica_id);
  	batch_sizes_[replica_id] = batch_size;
  }

  int GetBatchSize(int replica_id){
  	CHECK_GT(batch_sizes_.size(), replica_id);
  	return batch_sizes_[replica_id];
  }

  inline Dtype GetBatchSizeRatio(int device_id){
  	CHECK_EQ(batch_size_ratios_.count(device_id),1);
  	return batch_size_ratios_[device_id];
  }

  Dtype ForwardBackward(vector<Blob<Dtype>* >& bottom);
//  const vector<Blob<Dtype>*>& Forward(vector<Blob<Dtype>* >& bottom, Dtype *loss = NULL);
  void ForwardPrefilled(Dtype* loss);

  /// @brief Run forward using a set of bottom blobs, and return the result.
  const vector<Blob<Dtype>*>& Forward(const vector<Blob<Dtype>* > & bottom,
      Dtype* loss = NULL);
  /**
   * @brief Run forward using a serialized BlobProtoVector and return the
   *        result as a serialized BlobProtoVector
   */
  string Forward(const string& input_blob_protos, Dtype* loss = NULL);

  const shared_ptr<Blob<Dtype> > blob_by_name(const string& blob_name);
  /**
   * The network backward should take no input and output, since it solely
   * computes the gradient w.r.t the parameters, and the data has already been
   * provided during the forward pass.
   */
  void Backward();

  void CollectLoss();


  void CopyTrainedLayersFrom(const NetParameter& param);
  void CopyTrainedLayersFrom(const string trained_filename);
  /// @brief Writes the net to a proto.
  void ToProto(NetParameter* param, bool write_diff = false) const;

  DataManager<Dtype>* GetDataManager(){return data_manager_.get();}
	const vector<NetThread<Dtype>*>& GetNetThreads() const{
		return net_threads_;
	}

  // Helpers for Init.
  /**
   * @brief Remove layers that the user specified should be excluded given the current
   *        phase, level, and stage.
   */
  static void FilterNet(const NetParameter& param,
      NetParameter* param_filtered);
  /// @brief return whether NetState state meets NetStateRule rule
  static bool StateMeetsRule(const NetState& state, const NetStateRule& rule,
      const string& layer_name);

  void set_debug_info(const bool value);

  std::string name();

  /// @brief returns the parameters
  inline const vector<shared_ptr<Blob<Dtype> > >& params() const {
  	CHECK_GT(net_threads_.size(), 0);
    return net_threads_[0]->params();
  }

  /// @brief returns the parameter learning rate multipliers
  inline const vector<float>& params_lr() const {
  	CHECK_GT(net_threads_.size(), 0);
  	return net_threads_[0]->params_lr();
  }

  inline const vector<float>& params_weight_decay() const {
  	CHECK_GT(net_threads_.size(), 0);
  	return net_threads_[0]->params_weight_decay();
  }

  inline const vector<Blob<Dtype>*>& output_blobs() const {
    return net_output_blobs_;
  }

  /// @brief returns the blob names
  inline const vector<string>& blob_names() const{
  	CHECK_GT(net_threads_.size(), 0);
  	return net_threads_[0]->blob_names();
  }

  /// @brief returns the layers
  inline const vector<shared_ptr<Layer<Dtype> > >& layers() const {
  	CHECK_GT(net_threads_.size(), 0);
    return net_threads_[0]->layers();
  }
  /**
   * @brief returns the bottom vecs for each layer -- usually you won't
   *        need this unless you do per-layer checks such as gradients.
   */
  inline const vector<vector<Blob<Dtype>*> >& bottom_vecs() const {
  	CHECK_GT(net_threads_.size(), 0);
    return net_threads_[0]->bottom_vecs();
  }
  /**
   * @brief returns the top vecs for each layer -- usually you won't
   *        need this unless you do per-layer checks such as gradients.
   */
  inline const vector<vector<Blob<Dtype>*> >& top_vecs() const {
  	CHECK_GT(net_threads_.size(), 0);
    return net_threads_[0]->top_vecs();
  }
  inline const vector<vector<bool> >& bottom_need_backward() const {
  	CHECK_GT(net_threads_.size(), 0);
    return net_threads_[0]->bottom_need_backward();
  }

  inline const vector<Dtype>& blob_loss_weights() const {
  	CHECK_GT(net_threads_.size(), 0);
    return net_threads_[0]->blob_loss_weights();
  }

  inline const vector<int>& output_blob_indices() const {
  	CHECK_GT(net_threads_.size(), 0);
    return net_threads_[0]->output_blob_indices();
  }

  bool has_blob(const string& blob_name) const{
  	CHECK_GT(net_threads_.size(), 0);
    return net_threads_[0]->has_blob(blob_name);
  }

  void ComputeUpdateValue();
  /// @brief Updates the network weights based on the diff values computed.
  void Update();

  /**
   * @brief For an already initialized net, implicitly copies (i.e., using no
   *        additional memory) the pre-trained layers from another Net.
   */
  void ShareTrainedLayersWith(const Net<Dtype>* other);

  SolverParameter& get_solver_param(){return solver_param_;}

  Solver<Dtype> *get_solver(){return solver_;}

 protected:
  void ForwardBackwardHelper(const vector<Blob<Dtype>* >& bottom, Dtype *loss, bool do_backward);

  Solver<Dtype> *solver_;
  SolverParameter solver_param_;
  vector<int> device_ids_;
  vector<NetThread<Dtype>* > net_threads_;
  vector<map<int, shared_ptr<Layer<Dtype> > > > layer_map_;
  shared_ptr<DataManager<Dtype> > data_manager_;
  vector<Dtype> losses_;
  vector<Blob<Dtype>*> net_output_blobs_;
  int batch_size_;
  vector<int> batch_sizes_;
  // device id -> ratio
  std::map<int, Dtype> batch_size_ratios_;

  DISABLE_COPY_AND_ASSIGN(Net);
};


template <typename Dtype>
class NetThread : public InternalThread{
public:
  explicit NetThread(const NetParameter& param, int device_id, int replica_id,
  		Net<Dtype> *net, const SolverParameter &solver_param);
  virtual ~NetThread() {}

  inline int get_device_id(){return device_id_;}
  inline void set_bottom(vector<Blob<Dtype>* > &bottom){
  	for(int i = 0; i < bottom_.size(); ++i){
  		if(bottom_[i]){
    		delete bottom_[i];
  		}
  	}
  	bottom_ = bottom;
  }
  inline void set_input_blob_protos(std::string input_blob_protos){
  	input_blob_protos_ = input_blob_protos;
  }

  inline Dtype get_loss(){
  	return loss_;
  }

  inline Net<Dtype>* get_net(){
  	return net_;
  }


  void InitCuda();

  /// @brief Initialize a network with a NetParameter.
  void Init(const NetParameter& param);

  void PostInit();
  /**
   * @brief Run Forward with the input Blob%s already fed separately.
   *
   * You can get the input blobs using input_blobs().
   */
  const vector<Blob<Dtype>*>& ForwardPrefilled(Dtype* loss = NULL);

  /**
   * The From and To variants of Forward and Backward operate on the
   * (topological) ordering by which the net is specified. For general DAG
   * networks, note that (1) computing from one layer to another might entail
   * extra computation on unrelated branches, and (2) computation starting in
   * the middle may be incorrect if all of the layers of a fan-in are not
   * included.
   */
  Dtype ForwardFromTo(int start, int end);
  Dtype ForwardFrom(int start);
  Dtype ForwardTo(int end);
  /// @brief Run forward using a set of bottom blobs, and return the result.
  const vector<Blob<Dtype>*>& Forward(const vector<Blob<Dtype>* > & bottom,
      Dtype* loss = NULL);
  /**
   * @brief Run forward using a serialized BlobProtoVector and return the
   *        result as a serialized BlobProtoVector
   */
  string Forward(const string& input_blob_protos, Dtype* loss = NULL);

  /**
   * The network backward should take no input and output, since it solely
   * computes the gradient w.r.t the parameters, and the data has already been
   * provided during the forward pass.
   */
  void Backward();
  void BackwardFromTo(int start, int end);
  void BackwardFrom(int start);
  void BackwardTo(int end);

  /**
   * @brief Reshape all layers from bottom to top.
   *
   * This is useful to propagate changes to layer sizes without running
   * a forward pass, e.g. to compute output feature size.
   */
  void Reshape();

  void StartWork(){
    DLOG(INFO) << "CreateNetThread";
  	CreateNetThread();
  }

  void FinishWork(){
  	DLOG(INFO) << "JoinNetThread";
  	JoinNetThread();
  }

  Dtype GetLoss(){return loss_;}

  void ComputeUpdateValue();
  /// @brief Updates the network weights based on the diff values computed.
  void Update();

  /**
   * @brief For an already initialized net, implicitly copies (i.e., using no
   *        additional memory) the pre-trained layers from another Net.
   */
  void ShareTrainedLayersWith(const NetThread<Dtype>* other);
  // For an already initialized net, CopyTrainedLayersFrom() copies the already
  // trained layers from another net parameter instance.
  /**
   * @brief For an already initialized net, copies the pre-trained layers from
   *        another Net.
   */
  void CopyTrainedLayersFrom(const NetParameter& param);
  void CopyTrainedLayersFrom(const string trained_filename);
  /// @brief Writes the net to a proto.
  void ToProto(NetParameter* param, bool write_diff = false) const;

  /// @brief returns the network name.
  inline const string& name() const { return name_; }
  /// @brief returns the layer names
  inline const vector<string>& layer_names() const { return layer_names_; }
  /// @brief returns the blob names
  inline const vector<string>& blob_names() const { return blob_names_; }
  /// @brief returns the blobs
  inline const vector<shared_ptr<Blob<Dtype> > >& blobs() const {
    return blobs_;
  }
  /// @brief returns the layers
  inline const vector<shared_ptr<Layer<Dtype> > >& layers() const {
    return layers_;
  }
  /**
   * @brief returns the bottom vecs for each layer -- usually you won't
   *        need this unless you do per-layer checks such as gradients.
   */
  inline const vector<vector<Blob<Dtype>*> >& bottom_vecs() const {
    return bottom_vecs_;
  }
  /**
   * @brief returns the top vecs for each layer -- usually you won't
   *        need this unless you do per-layer checks such as gradients.
   */
  inline const vector<vector<Blob<Dtype>*> >& top_vecs() const {
    return top_vecs_;
  }
  inline const vector<vector<bool> >& bottom_need_backward() const {
    return bottom_need_backward_;
  }
  inline const vector<Dtype>& blob_loss_weights() const {
    return blob_loss_weights_;
  }
  /// @brief returns the parameters
  inline vector<shared_ptr<Blob<Dtype> > >& params() {
    return params_;
  }

  inline const vector<shared_ptr<BlobSolver<Dtype> > >& params_solver(){
  	return params_solver_;
  }
  /// @brief returns the parameter learning rate multipliers
  inline const vector<float>& params_lr() const { return params_lr_; }
  inline const vector<float>& params_weight_decay() const {
    return params_weight_decay_;
  }
  inline const vector<int>& params_shard_size(){return params_shard_size_;}

  const map<string, int>& param_names_index() const {
    return param_names_index_;
  }
  /// @brief Input and output blob numbers
  inline int num_inputs() const { return net_input_blobs_.size(); }
  inline int num_outputs() const { return net_output_blobs_.size(); }
  inline const vector<Blob<Dtype>*>& input_blobs() const {
    return net_input_blobs_;
  }
  inline const vector<Blob<Dtype>*>& output_blobs() const {
    return net_output_blobs_;
  }
  inline const vector<int>& input_blob_indices() const {
    return net_input_blob_indices_;
  }
  inline const vector<int>& output_blob_indices() const {
    return net_output_blob_indices_;
  }
  bool has_blob(const string& blob_name) const;
  const shared_ptr<Blob<Dtype> > blob_by_name(const string& blob_name) const;
  bool has_layer(const string& layer_name) const;
  const shared_ptr<Layer<Dtype> > layer_by_name(const string& layer_name) const;

  void set_debug_info(const bool value) { debug_info_ = value; }

  void set_work_message(WorkMessage work_message){
  	work_message_ = work_message;
  }

  Solver<Dtype> * GetExternalSolver(){return net_->get_solver();}

  int get_replica_id(){return replica_id_;}

  void add_replicas(NetThread<Dtype> *nt){
  	CHECK_EQ(replicas_.count(nt->get_replica_id()), 0);
  	replicas_[nt->get_replica_id()] = nt;
  }

  std::map<int, NetThread<Dtype>* >& get_replicas(){
  	return replicas_;
  }

//  shared_ptr<Blob<Dtype> > GetShardGPUOnly(const shared_ptr<Blob<Dtype> > &p, int param_id, int replica_id);

  shared_ptr<Blob<Dtype> > GetShardGPUOnly(int param_id, int replica_id);


//  shared_ptr<BlobDiffReducer<Dtype> > get_blob_diff_reducer(){
//  	return blob_diff_reducer_;
//  }

//  shared_ptr<IBroadcastDiffNetwork<Dtype> > get_blob_diff_broadcaster(){
//  	return blob_diff_broadcaster_;
//  }


 protected:
  // Helpers for Init.
  /// @brief Append a new input or top blob to the net.
  void AppendTop(const NetParameter& param, const int layer_id,
                 const int top_id, set<string>* available_blobs,
                 map<string, int>* blob_name_to_idx);
  /// @brief Append a new bottom blob to the net.
  int AppendBottom(const NetParameter& param, const int layer_id,
                   const int bottom_id, set<string>* available_blobs,
                   map<string, int>* blob_name_to_idx);
  /// @brief Append a new parameter blob to the net.
  void AppendParam(const NetParameter& param, const int layer_id,
                   const int param_id);

  /// @brief Helper for displaying debug info in Forward about input Blobs.
  void InputDebugInfo(const int layer_id);
  /// @brief Helper for displaying debug info in Forward.
  void ForwardDebugInfo(const int layer_id);
  /// @brief Helper for displaying debug info in Backward.
  void BackwardDebugInfo(const int layer_id);
  /// @brief Helper for displaying debug info in Update.
  void UpdateDebugInfo(const int param_id);

  /// @brief Get misc parameters, e.g. the LR multiplier and weight decay.
  void GetLearningRateAndWeightDecay();

  /// @brief Individual layers in the net
  vector<shared_ptr<Layer<Dtype> > > layers_;
  vector<string> layer_names_;
  map<string, int> layer_names_index_;
  vector<bool> layer_need_backward_;
  /// @brief the blobs storing intermediate results between the layer.
  vector<shared_ptr<Blob<Dtype> > > blobs_;
  vector<string> blob_names_;
  map<string, int> blob_names_index_;
  vector<bool> blob_need_backward_;
  /// bottom_vecs stores the vectors containing the input for each layer.
  /// They don't actually host the blobs (blobs_ does), so we simply store
  /// pointers.
  vector<vector<Blob<Dtype>*> > bottom_vecs_;
  vector<vector<int> > bottom_id_vecs_;
  vector<vector<bool> > bottom_need_backward_;
  /// top_vecs stores the vectors containing the output for each layer
  vector<vector<Blob<Dtype>*> > top_vecs_;
  vector<vector<int> > top_id_vecs_;
  /// Vector of weight in the loss (or objective) function of each net blob,
  /// indexed by blob_id.
  vector<Dtype> blob_loss_weights_;
  vector<vector<int> > param_id_vecs_;
  vector<int> param_owners_;
  vector<string> param_display_names_;
  vector<pair<int, int> > param_layer_indices_;
  map<string, int> param_names_index_;
  /// blob indices for the input and the output of the net
  vector<int> net_input_blob_indices_;
  vector<int> net_output_blob_indices_;
  vector<Blob<Dtype>*> net_input_blobs_;
  vector<Blob<Dtype>*> net_output_blobs_;
  string name_;
  /// The parameters in the network.
  vector<shared_ptr<Blob<Dtype> > > params_;
  vector<shared_ptr<BlobSolver<Dtype> > > params_solver_;

  vector<int> params_shard_size_;
  /// the learning rate multipliers
  vector<float> params_lr_;
  /// the weight decay multipliers
  vector<float> params_weight_decay_;
  /// The bytes of memory used by this net
  size_t memory_used_;
  /// Whether to compute and display debug info for the net.
  bool debug_info_;
  /// if it's a testing net
  bool test_net_;

	virtual void CreateNetThread();
	virtual void JoinNetThread();
  virtual void InternalThreadEntry();

//  shared_ptr<NetThreadSolver<Dtype> > solver_;
  int device_id_;
  int replica_id_;
  std::map<int, NetThread<Dtype>* > replicas_;
//  const vector<Blob<Dtype>* > *bottom_;
  vector<Blob<Dtype>* > bottom_;
  std::string input_blob_protos_;
  Dtype loss_;
  Net<Dtype>* net_;
  WorkMessage work_message_;
//  shared_ptr<BlobDiffReducer<Dtype> > blob_diff_reducer_;
//  shared_ptr<IBroadcastDiffNetwork<Dtype> > blob_diff_broadcaster_;

  boost::mutex params_mutex_;


  DISABLE_COPY_AND_ASSIGN(NetThread);
};

}  // namespace caffe

#endif  // CAFFE_NET_HPP_
