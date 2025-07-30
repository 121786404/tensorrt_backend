#pragma once
#include "triton/backend/backend_model_instance.h"
#include<vector>
#include"ixrt_model.h"
#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "NvInferRuntimeCommon.h"
#include "memory_utils.h"
#include "logging.h"
#include <iostream>
#include <unordered_set>
#include <unordered_map>

namespace triton { namespace backend { namespace ixrt {
class ModelInstanceState : public BackendModelInstance {
public:
  static TRITONSERVER_Error* Create(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance,
      ModelInstanceState** state);
  ~ModelInstanceState();

  bool SupportsDynamicBatching();

  void ProcessRequests(
      TRITONBACKEND_Request** requests, const uint32_t request_count);

  // Get the state of the model that corresponds to this instance.
  ModelState* StateForModel() const { return model_state_; }

 private:
  ModelState* model_state_;

  bool supports_first_dim_batching_ = false; 
  bool dynamic_engine_ = false;
  UniquePtr<nvinfer1::ICudaEngine> engine_;
  std::vector<void*> binding_buffers_;

  int cuda_stream_priority_{0};

  cudaStream_t input_copy_stream_{};
  cudaStream_t output_copy_stream_{};


  UniquePtr<nvinfer1::IExecutionContext> context_;
  std::unordered_map<std::string, int32_t> model_inputs_;
  std::unordered_map<std::string, int32_t> model_outputs_;
  struct IOAttribute {
    std::string name;
    std::string data_type;
    std::vector<int32_t> dims;
  };

  std::vector<IOAttribute> config_inputs_;
  std::vector<IOAttribute> config_outputs_;

  std::unordered_map<int32_t, UniArrPtr<int8_t>> output_host_buffers_;
  Logger logger_;
  int64_t max_batch_size_;


  ModelInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance)
      : BackendModelInstance(model_state, triton_model_instance),
        model_state_(model_state), logger_(nvinfer1::ILogger::Severity::kERROR)
  {
  }

  void InitBindingBuffers();
  void InitFixedIOBuffers();
  void InitDynamicIOBuffers();
  void LoadModule(const std::string& file_name, const int& device_id);
  void InitConfigIO(const char* name, std::vector<IOAttribute>* ios);
  void CheckConfig();

  TRITONSERVER_DataType convertStrDataType2Datatype(const std::string& strDataType);   
};  
}}}
