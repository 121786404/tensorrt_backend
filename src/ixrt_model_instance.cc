#include"ixrt_model_instance.h"
#include "triton/common/logging.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_output_responder.h"
#include "NvInferPlugin.h"

using namespace nvinfer1;
namespace triton { namespace backend { namespace ixrt {
std::unordered_map<std::string, DataType> ModelConfigDataType {
    {"TYPE_INT8", DataType::kINT8},
    {"TYPE_FP16", DataType::kHALF},
    {"TYPE_FP32", DataType::kFLOAT},
    {"TYPE_INT32", DataType::kINT32},
    {"TYPE_INT64", DataType::kINT64},
    {"TYPE_BOOL", DataType::kBOOL},
    {"TYPE_UINT8", DataType::kUINT8}
};

std::unordered_map<DataType, std::string> DataTypeStr {
    {DataType::kINT8, "TYPE_INT8"},
    {DataType::kHALF, "TYPE_FP16"},
    {DataType::kFLOAT, "TYPE_FP32"},
    {DataType::kINT32, "TYPE_INT32"},
    {DataType::kINT64, "TYPE_INT64"},
    {DataType::kBOOL, "TYPE_BOOL"},
    {DataType::kUINT8, "TYPE_UINT8"}
};

void LoadBufferFromDisk(const std::string& file_path, std::vector<int8_t>* engine_buffer) {
    std::ifstream in_file(file_path, std::ios::binary);
    if (not in_file.is_open()) {
        LOG_ERROR << "Could not load engine file from " << file_path << std::endl;
          LOG_MESSAGE(
            TRITONSERVER_LOG_ERROR,
            (std::string("Could not load engine file from ") + file_path).c_str());
        in_file.close();
        return;
    }
    in_file.seekg(0, std::ios::end);
    uint64_t file_length = in_file.tellg();
    in_file.seekg(0, std::ios::beg);
    engine_buffer->resize(file_length);
    in_file.read((char*)engine_buffer->data(), file_length);
    in_file.close();
    LOG_INFO << "Load buffer size " << file_length << std::endl;
    LOG_MESSAGE(
            TRITONSERVER_LOG_INFO,
            (std::string("Load buffer size ") + std::to_string(file_length)).c_str());
}

bool ModelInstanceState::SupportsDynamicBatching() {
  return supports_first_dim_batching_;
}

void ModelInstanceState::LoadModule(const std::string& file_name, const int& device_id) {
  std::vector<int8_t> engine_buffer;
  LoadBufferFromDisk(file_name, &engine_buffer);
  cudaSetDevice(device_id);
  initLibNvInferPlugins(&logger_, "");

  UniquePtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger_)};
  if (not runtime) {
      LOG_INFO << "Create ixrt runtime failed" << std::endl;
      return;
  } else {
      LOG_INFO << "Create ixrt runtime done" << std::endl;
  }
  engine_ = UniquePtr<nvinfer1::ICudaEngine>(
      runtime->deserializeCudaEngine(engine_buffer.data(), engine_buffer.size()));
  context_ = UniquePtr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
  auto num_bindings = engine_->getNbBindings();
  for (auto i = 0; i < num_bindings; ++i) {
    if (engine_->bindingIsInput(i)) {
      model_inputs_.emplace(engine_->getBindingName(i), i);
    } else {
      model_outputs_.emplace(engine_->getBindingName(i), i);
    }
  }
  for (const auto& [name, index] : model_inputs_) {
    auto dim = engine_->getBindingDimensions(index);
    for (auto i = 0; i < dim.nbDims; ++i) {
      if (dim.d[i] < 0) {
        dynamic_engine_ = true;
      }
    }
  }
}

void ModelInstanceState::CheckConfig() {
  // Only check input dims
  if (max_batch_size_ > 0 and (not dynamic_engine_)) {
    LOG_MESSAGE(
            TRITONSERVER_LOG_ERROR,
            std::string("Use dynamic batching must use engine with dynamic shape").c_str());
    THROW_IF_BACKEND_INSTANCE_ERROR(TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNAVAILABLE,
          std::string("Use dynamic batching must use engine with dynamic shape").c_str()));
  } else {
    LOG_MESSAGE(
            TRITONSERVER_LOG_INFO,
            std::string("Use fixed batching from model config").c_str());
  }
  for (auto& io_attr : config_inputs_) {
    if (not model_inputs_.count(io_attr.name)) {
      LOG_MESSAGE(
            TRITONSERVER_LOG_ERROR,
            std::string("Config input name : " + io_attr.name + " is not a valid input interface").c_str());
        THROW_IF_BACKEND_INSTANCE_ERROR(TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNAVAILABLE,
          std::string("Config input name : " + io_attr.name + " is not a valid input interface").c_str()));
    } else {
      LOG_MESSAGE(
            TRITONSERVER_LOG_INFO,
            std::string("Confirmed input name: " + io_attr.name).c_str());
    }
    if (engine_->getBindingDataType(model_inputs_.at(io_attr.name)) != ModelConfigDataType.at(io_attr.data_type)) {
      LOG_MESSAGE(
            TRITONSERVER_LOG_ERROR,
            std::string("Config input name : " + io_attr.name + ", with type: " + io_attr.data_type 
              + " is not consistent with model input type: " + DataTypeStr.at(engine_->getBindingDataType(model_inputs_.at(io_attr.name)))).c_str());
      THROW_IF_BACKEND_INSTANCE_ERROR(TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNAVAILABLE,
          std::string("Config input name : " + io_attr.name + ", with type: " + io_attr.data_type 
          + " is not consistent with model input type" ).c_str()));
    } else {
      LOG_MESSAGE(
            TRITONSERVER_LOG_INFO,
            std::string("Confirmed input name: " + io_attr.name + " data type: " + io_attr.data_type).c_str());
    }
    auto model_dims = engine_->getBindingDimensions(model_inputs_.at(io_attr.name));
    //printf("model dims size: %d, config dims size: %d\n", model_dims.nbDims, io_attr.dims.size());
    if(dynamic_engine_ && model_dims.nbDims == 1 && io_attr.dims.size() > 1) {
      LOG_MESSAGE(
            TRITONSERVER_LOG_INFO,
            std::string("The input dimension of the dynamic engine must be at least 1").c_str());
      io_attr.dims.pop_back();
    }
    if (model_dims.nbDims != io_attr.dims.size()) {
      LOG_MESSAGE(
            TRITONSERVER_LOG_ERROR,
            std::string("Config input name : " + io_attr.name + ", with dim size: " + std::to_string(io_attr.dims.size())
              + " is not consistent with model input size " + std::to_string(model_dims.nbDims) ).c_str());
      THROW_IF_BACKEND_INSTANCE_ERROR(TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNAVAILABLE,
          std::string("Config input name : " + io_attr.name + ", with dim size: " + std::to_string(io_attr.dims.size())
          + " is not consistent with model input size " + std::to_string(model_dims.nbDims) ).c_str()));
    } else {
            LOG_MESSAGE(
            TRITONSERVER_LOG_INFO,
            std::string("Confirmed input name: " + io_attr.name + " dim size: " + std::to_string(io_attr.dims.size())).c_str());
    }
  }
}

void ModelInstanceState::InitFixedIOBuffers() {

  auto num_bindings = engine_->getNbBindings();
  binding_buffers_.resize(num_bindings, nullptr);
  for (auto i = 0; i < num_bindings; ++i) {
    auto dims = engine_->getBindingDimensions(i);
    auto name = engine_->getBindingName(i);
    // printf("The bindings %d, type %d\n", i, engine_->getBindingDataType(i));
    auto buffer_size = GetBytes(dims, engine_->getBindingDataType(i));
    CHECK(cudaMalloc(&binding_buffers_.at(i), buffer_size));
    LOG_MESSAGE(
            TRITONSERVER_LOG_INFO,
            (std::string("Allocate binding buffer ") + std::string(name) + ", size " + std::to_string(buffer_size)).c_str());
    if (not engine_->bindingIsInput(i)) {
      output_host_buffers_.emplace(i, UniArrPtr<int8_t>(new int8_t[buffer_size]));
    }
  }
}
void ModelInstanceState::InitDynamicIOBuffers() {
    auto num_bindings = engine_->getNbBindings();
    binding_buffers_.resize(num_bindings, nullptr);
    auto profile_idx = context_->getOptimizationProfile();
    
    std::vector<std::pair<int32_t, nvinfer1::Dims>> input_dims; 
    for (auto i = 0; i < num_bindings; ++i) {
      if (not engine_->bindingIsInput(i)) {
        continue;
      }
      auto input_dim = engine_->getProfileDimensions(i, profile_idx, nvinfer1::OptProfileSelector::kMAX);
      context_->setBindingDimensions(i, input_dim);

    }

    for (auto i = 0; i < num_bindings; ++i) {
      auto binding_dim = context_->getBindingDimensions(i);
      auto name = engine_->getBindingName(i);
      if (binding_dim.d[0] < max_batch_size_) {
        LOG_MESSAGE(
            TRITONSERVER_LOG_ERROR,
            std::string("Model input name : " + std::string(name) + ", max input batch: " + std::to_string(binding_dim.d[0])
              + " config max batching is: " + std::to_string(max_batch_size_) ).c_str());
        THROW_IF_BACKEND_INSTANCE_ERROR(TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNAVAILABLE,
          std::string("Model input name : " + std::string(name) + ", max input batch: " + std::to_string(binding_dim.d[0])
          + " config max batching is: " + std::to_string(max_batch_size_) ).c_str()));
      }
      
      // printf("The %d binding, name %s\n", i, name);
      // for (auto k = 0; k < binding_dim.nbDims; ++k) {
      //   printf("%d,", binding_dim.d[k]);

      // }
      // printf("\n");
      auto buffer_size = GetBytes(binding_dim, engine_->getBindingDataType(i));
      CHECK(cudaMalloc(&binding_buffers_.at(i), buffer_size));
      LOG_MESSAGE(
            TRITONSERVER_LOG_INFO,
            (std::string("Allocate binding buffer ") + std::string(name) + ", size " + std::to_string(buffer_size)).c_str());
      if (not engine_->bindingIsInput(i)) {
        output_host_buffers_.emplace(i, UniArrPtr<int8_t>(new int8_t[buffer_size]));
      }
    }
}

void ModelInstanceState::InitConfigIO(const char* name, std::vector<IOAttribute>* ios) {
  ModelState* model_state = reinterpret_cast<ModelState*>(StateForModel());
  triton::common::TritonJson::Value config_model_io;  
  model_state->config_.MemberAsArray(name, &config_model_io);
  size_t num_ios = config_model_io.ArraySize();
  for (auto i = 0; i < num_ios; ++i) {
    triton::common::TritonJson::Value model_io_item;  
    config_model_io.IndexAsObject(i, &model_io_item);
    std::string name;
    model_io_item.MemberAsString("name", &name);
    // std::cout << "name:" << name << std::endl;
    std::string data_type;
    model_io_item.MemberAsString("data_type", &data_type);
    // std::cout << "data_type:" << data_type << std::endl;

    if (not ModelConfigDataType.count(data_type)) {
      LOG_MESSAGE(
        TRITONSERVER_LOG_ERROR,
        ("not supported data type:" + data_type).c_str());
    }
    IOAttribute cache;
    cache.name = name;
    cache.data_type = data_type;
    if(SupportsDynamicBatching()) {
      cache.dims.emplace_back(-1);
    }
    triton::common::TritonJson::Value dims;
    model_io_item.MemberAsArray("dims", &dims);
    for (int j = 0; j < dims.ArraySize(); j++) {
      triton::common::TritonJson::Value dim_channel;
      dims.At(j, &dim_channel);
      int64_t dim_channel_int64;
      dim_channel.AsInt(&dim_channel_int64);
      //std::cout << dim_channel_int64 << " ";
      // printf("%d\t", dim_channel_int64);
      cache.dims.emplace_back(dim_channel_int64);
    }
    // printf("\n");
    ios->emplace_back(cache);
  }
}
void ModelInstanceState::InitBindingBuffers() {
  ModelState* model_state = reinterpret_cast<ModelState*>(StateForModel());
  model_state->SupportsFirstDimBatching(&supports_first_dim_batching_);
  model_state->config_.MemberAsInt("max_batch_size", &max_batch_size_);
  // printf("max batch size: %d\n", max_batch_size_);
  InitConfigIO("input", &config_inputs_);
  InitConfigIO("output", &config_outputs_);
  CheckConfig();

  if (dynamic_engine_) {
    // printf("Allocate binding buffer for dynamic IO\n");
    InitDynamicIOBuffers();
  } else {
    // printf("Allocate FixIO buffer for normal model\n");
    InitFixedIOBuffers();
  }
}

TRITONSERVER_Error*
ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state)
{
  try {
    *state = new ModelInstanceState(model_state, triton_model_instance);
  }
  catch (const BackendModelInstanceException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelInstanceException"));
    RETURN_IF_ERROR(ex.err_);
  }

  // is this model support dynamic batching?
  if ((*state)->SupportsDynamicBatching()) { 
    LOG_MESSAGE(
            TRITONSERVER_LOG_INFO,
            (model_state->Name() + std::string(" support dynamic-batching, max batch size is ") + 
            std::to_string(model_state->MaxBatchSize())).c_str());
  } else {
    LOG_MESSAGE(
            TRITONSERVER_LOG_INFO,
            std::string(model_state->Name() + " is non-batching model, all input shape are frozen.").c_str());
  }
  std::string model_filename;
  model_state->config_.MemberAsString("default_model_filename", &model_filename);
  
  auto model_path = JoinPath(
      {model_state->RepositoryPath(), std::to_string(model_state->Version()),
       model_filename});
  LOG_MESSAGE(
            TRITONSERVER_LOG_INFO,
            (std::string("Load model from path ") + model_path).c_str());

  std::string str;
  str = std::to_string((*state)->DeviceId());
  //LOG_MESSAGE(TRITONSERVER_LOG_INFO, ("DeviceId: " + std::to_string(DeviceId())).c_str());
  LOG_MESSAGE(TRITONSERVER_LOG_INFO, ("DeviceId: " + str).c_str());

  (*state)->LoadModule(model_path, (*state)->DeviceId());
  (*state)->InitBindingBuffers();
  
  // common::TritonJson::Value instance_group;
  // model_state->config_.Find("instance_group", &instance_group);

  return nullptr;  // success
}

ModelInstanceState::~ModelInstanceState() {
  LOG_MESSAGE(
            TRITONSERVER_LOG_INFO,
            std::string("Release GPU memory for bindings size " + std::to_string(binding_buffers_.size())).c_str());
  for (auto& buffer : binding_buffers_) {
    CHECK(cudaFree(buffer));
  }
}

TRITONSERVER_DataType ModelInstanceState::convertStrDataType2Datatype(const std::string& strDataType){
  if(strDataType.compare("TYPE_BOOL")==0){
    return TRITONSERVER_TYPE_BOOL;
  }
  if(strDataType.compare("TYPE_UINT8")==0){
    return TRITONSERVER_TYPE_UINT8;
  }
  if(strDataType.compare("TYPE_UINT16")==0){
    return TRITONSERVER_TYPE_UINT16;
  }
  if(strDataType.compare("TYPE_UINT32")==0){
    return TRITONSERVER_TYPE_UINT32;
  }
  if(strDataType.compare("TYPE_UINT64")==0){
    return TRITONSERVER_TYPE_UINT64;
  }
  if(strDataType.compare("TYPE_INT8")==0){
    return TRITONSERVER_TYPE_INT8;
  }
  if(strDataType.compare("TYPE_INT16")==0){
    return TRITONSERVER_TYPE_INT16;
  }
  if(strDataType.compare("TYPE_INT32")==0){
    return TRITONSERVER_TYPE_INT32;
  }
  if(strDataType.compare("TYPE_INT64")==0){
    return TRITONSERVER_TYPE_INT64;
  }
  if(strDataType.compare("TYPE_FP16")==0){
    return TRITONSERVER_TYPE_FP16;
  }
  if(strDataType.compare("TYPE_FP32")==0){
    return TRITONSERVER_TYPE_FP32;
  }
  if(strDataType.compare("TYPE_FP64")==0){
    return TRITONSERVER_TYPE_FP64;
  }
  if(strDataType.compare("TYPE_FP16")==0){
    return TRITONSERVER_TYPE_FP16;
  }
  if(strDataType.compare("TYPE_BYTES")==0){
    return TRITONSERVER_TYPE_BYTES;
  }
  if(strDataType.compare("TYPE_BF16")==0){
    return TRITONSERVER_TYPE_BF16;
  }
  
  return TRITONSERVER_TYPE_INVALID;
}


void ModelInstanceState::ProcessRequests(
      TRITONBACKEND_Request** requests, const uint32_t request_count) {

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("TRITONBACKEND_ModelExecute: Running ") + Name() + " with " +
       std::to_string(request_count) + " requests")
          .c_str());

  const int max_batch_size = model_state_->MaxBatchSize();
  LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, ("max_batch_size: " + std::to_string(max_batch_size)).c_str());
  
  cudaSetDevice(DeviceId());

  size_t total_batch_size = 0;  

  uint64_t exec_start_ns = 0;
  SET_TIMESTAMP(exec_start_ns);
  for (size_t i = 0; i < request_count; i++) {
    // If we get a nullptr request then something is badly wrong. Fail
    // and release all requests.
    if (requests[i] == nullptr) {
      LOG_MESSAGE(
        TRITONSERVER_LOG_ERROR,
            std::string("null request given to IXRT backend for '" + Name() + "'").c_str());
      RequestsRespondWithError(
          requests, request_count,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              std::string(
                  "null request given to IXRT backend for '" + Name() + "'")
                  .c_str()));
    }
  }
  uint64_t receiver_start_ns = 0;
  SET_TIMESTAMP(receiver_start_ns);

  std::vector<TRITONBACKEND_Response*> responses;
  responses.reserve(request_count);

  for (size_t i = 0; i < request_count; i++) {
    TRITONBACKEND_Response* response;
    auto err = TRITONBACKEND_ResponseNew(&response, requests[i]);
    if (err == nullptr) {
      responses.emplace_back(response);
    } else {
      responses.emplace_back(nullptr);
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "Fail to create response");
      TRITONSERVER_ErrorDelete(err);
    }
  }


  for (size_t i = 0; i < request_count; i++) {
    if (max_batch_size > 0) {
      // Retrieve the batch size from one of the inputs, if the model
      // supports batching, the first dimension size is batch size.
      TRITONBACKEND_Input* input;
      TRITONSERVER_Error* err =
          TRITONBACKEND_RequestInputByIndex(requests[i], 0 /* index */, &input);
      if (err == nullptr) {
        const int64_t* shape;
        err = TRITONBACKEND_InputProperties(
            input, nullptr, nullptr, &shape, nullptr, nullptr, nullptr);
        total_batch_size += shape[0];
      }
      if (err != nullptr) {
      RESPOND_ALL_AND_SET_NULL_IF_ERROR(
        responses, request_count, TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNAVAILABLE,
          std::string("Failed to get input properties").c_str()));
      }
    } else {
      total_batch_size += 1;
    }
  }

  // If there are no valid payloads then no need to run the inference.
  if (total_batch_size == 0) {
    return;
  }

  ModelState* model_state = reinterpret_cast<ModelState*>(StateForModel());

  uint32_t req_input_count;
  TRITONBACKEND_RequestInputCount(*requests, &req_input_count);

  CreateCudaStream(DeviceId(), 0, &output_copy_stream_);
  CreateCudaStream(DeviceId(), 0, &input_copy_stream_);

  std::vector<const char*> input_names;

  BackendInputCollector collector(
      requests, request_count, &responses, model_state->TritonMemoryManager(),
      false /* pinned_enabled */, input_copy_stream_ /* stream*/, nullptr, nullptr, 0, nullptr, true, true);

  // std::vector<std::pair<TRITONSERVER_MemoryType, int64_t>> allowed_input_types =
  //     {{TRITONSERVER_MEMORY_CPU_PINNED, 0}, {TRITONSERVER_MEMORY_CPU, 0}, {TRITONSERVER_MEMORY_GPU, 0}};

  TRITONSERVER_MemoryType buffer_memory_type = TRITONSERVER_MEMORY_GPU;//input_buffer_memory_type;
  int64_t buffer_memory_type_id = DeviceId(); //input_buffer_memory_type_id;

  uint32_t offset = 0;
  
  for ( size_t idx = 0; idx < req_input_count; idx++) {

    TRITONBACKEND_Input* input;
    TRITONBACKEND_RequestInputByIndex(requests[0], idx, &input);    

    // TRITONBACKEND_Input *req_input;
    // TRITONBACKEND_RequestInput(*requests, req_input_name, &req_input);

    const char* req_input_name;
    TRITONSERVER_DataType datatype;
    const int64_t* shape;
    uint32_t dims_count;
    TRITONBACKEND_InputProperties(input, &req_input_name, &datatype, &shape, &dims_count, nullptr, nullptr);
    
    input_names.emplace_back(req_input_name);
  
    
    // printf("name %s, type: %d, dim size: %d\n", req_input_name, datatype, dims_count);
    // printf("shape : %d, %d, %d, %d\n", shape[0] ,shape[1] ,shape[2], shape[3]);
    std::vector<int64_t> batchn_shape(
      shape, shape + dims_count);

    if(max_batch_size > 0){
      batchn_shape[0] = total_batch_size;
    }
    
    Dims input_dim;
    input_dim.nbDims = dims_count;
    for (auto i = 0; i < dims_count; ++i) {
      input_dim.d[i] = batchn_shape[i];
    }

    auto bind_idx = engine_->getBindingIndex(req_input_name);
    auto set_status = context_->setBindingDimensions(bind_idx, input_dim);
    auto engine_dim = context_->getBindingDimensions(bind_idx);
    
    if ((not set_status) and dynamic_engine_) {
      LOG_MESSAGE(
        TRITONSERVER_LOG_ERROR,
        std::string("Set bind dimension for  " + std::to_string(bind_idx) + " is not accepted, named: " 
          + std::string(engine_->getBindingName(bind_idx)) ).c_str());
        RESPOND_ALL_AND_SET_NULL_IF_ERROR(
        responses, request_count, TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNAVAILABLE,
          std::string("Set bind dimension for  " + std::to_string(bind_idx) + " is not accepted, named: " 
          + std::string(engine_->getBindingName(bind_idx)) ).c_str()));
      return;
    }

    // printf("Set binding dimension\n");
    //const char* input_buffer = (char*)binding_buffers_.at(bind_idx);
    const char* input_buffer;
    size_t input_byte_size = GetBytes(engine_dim, engine_->getBindingDataType(bind_idx));
    size_t input_buffer_byte_size;


    // RESPOND_ALL_AND_SET_NULL_IF_ERROR(
    //     responses, request_count,
    //     collector.ProcessTensor(
    //         req_input_name, nullptr /* existing_buffer */,
    //         0 /* existing_buffer_byte_size */, allowed_input_types, &input_buffer,
    //         &input_buffer_byte_size, &buffer_memory_type,
    //         &buffer_memory_type_id/*0-CPU, 1-CPU_PINNED, 2-GPU*/));

    collector.ProcessTensor(
        req_input_name,  static_cast<char*>(binding_buffers_.at(bind_idx))+ offset,  input_byte_size,
        buffer_memory_type, buffer_memory_type_id);
  
    offset += input_byte_size;

    cudaStreamSynchronize(input_copy_stream_);

    const bool need_cuda_input_sync = collector.Finalize();
    // if (need_cuda_input_sync) {
    //   LOG_MESSAGE(
    //       TRITONSERVER_LOG_ERROR,
    //       "'ixrt' backend: unexpected CUDA sync required by collector");
    // }
    // If there is batching model, server will receive buffer smaller than max capacity of model input
    // if there is a non-batching model, client request data size in bytes should exactly equal to model's inputs.
    //LOG_VERBOSE(1) << "input_buffer_byte_size: " << input_buffer_byte_size << "\n";


  }
  uint64_t receiver_end_ns = 0;
  SET_TIMESTAMP(receiver_end_ns);

  uint64_t compute_start_ns = 0;
  SET_TIMESTAMP(compute_start_ns);
  // printf("Execute\n");
  // after all input are ready, we do the model inference.
  context_->executeV2(binding_buffers_.data());
  // printf("execute done\n");
  uint64_t compute_end_ns = 0;
  SET_TIMESTAMP(compute_end_ns);

  uint64_t responder_start_ns = 0;
  SET_TIMESTAMP(responder_start_ns);
  BackendOutputResponder responder(
      requests, request_count, &responses, model_state->TritonMemoryManager(),
      false, false /* pinned_enabled */,
      output_copy_stream_ /* stream*/, nullptr, true);
  
  std::map<std::string, std::string> mapOutputAttr;
  for(std::vector<IOAttribute>::iterator it = config_outputs_.begin(); it != config_outputs_.end(); ++it){
    mapOutputAttr[it->name] = it->data_type;
  }

  uint32_t req_output_count;
  TRITONBACKEND_RequestOutputCount(*requests, &req_output_count);

  uint32_t output_offset = 0;
  for (auto i = 0; i < req_output_count; ++i) {
    const char* req_output_name;
    TRITONBACKEND_RequestOutputName(*requests, i, &req_output_name);

    TRITONBACKEND_Input* input;
    TRITONBACKEND_RequestInputByIndex(requests[0], i, &input);    

    // TRITONBACKEND_Input *req_input;
    // TRITONBACKEND_RequestInput(*requests, req_input_name, &req_input);

    const int64_t* shape;
    TRITONBACKEND_InputProperties(input, nullptr, nullptr, &shape, nullptr, nullptr, nullptr);

    if (not model_outputs_.count(req_output_name)) {
      continue;
    }
    // printf("Output name %s\n", req_output_name);
    auto output_idx =  engine_->getBindingIndex(req_output_name);
    auto dims = context_->getBindingDimensions(output_idx);
    std::vector<int64_t> batchn_shape {};
    for (int k=0; k< dims.nbDims; k++){
      batchn_shape.push_back(dims.d[k]);
    }

    batchn_shape[0] = shape[0];
    // printf("batchn_shape[0]: %d\n", batchn_shape[0]);
    dims.d[0] = shape[0];
    auto output_byte_size = GetBytes(dims, engine_->getBindingDataType(output_idx));

    TRITONSERVER_DataType dataType = TRITONSERVER_TYPE_FP32;
    if (mapOutputAttr.find(req_output_name) != mapOutputAttr.end()) {
        std::string strDataType = mapOutputAttr[req_output_name];
        dataType = convertStrDataType2Datatype(strDataType);
        if (dataType == TRITONSERVER_TYPE_INVALID) {
            dataType = TRITONSERVER_TYPE_FP32;
        }
    }

    responder.ProcessTensor(
      req_output_name, dataType, batchn_shape, (char*) (binding_buffers_.at(output_idx)) + output_offset,
      buffer_memory_type, buffer_memory_type_id);

    output_offset += output_byte_size;

    cudaStreamSynchronize(output_copy_stream_);
    const bool need_cuda_output_sync = responder.Finalize();

  }

  // Send all the responses that haven't already been sent because of
  // an earlier error.
  for (auto& response : responses) {
      if (response != nullptr) {
        LOG_IF_ERROR(
            TRITONBACKEND_ResponseSend(
                response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
            "failed to send response");
    }
  }

  uint64_t responder_end_ns = 0;
  SET_TIMESTAMP(responder_end_ns);

  uint64_t exec_end_ns = 0;
  SET_TIMESTAMP(exec_end_ns);

#ifdef TRITON_ENABLE_STATS
  // For batch statistics need to know the total batch size of the
  // requests. This is not necessarily just the number of requests,
  // because if the model supports batching then any request can be a
  // batched request itself.
  // size_t total_batch_size = 0;
  // if (not supports_first_dim_batching_) {
  //   total_batch_size = request_count;
  // } else {
  //   for (uint32_t r = 0; r < request_count; ++r) {
  //     auto& request = requests[r];
  //     TRITONBACKEND_Input* input = nullptr;
  //     LOG_IF_ERROR(
  //         TRITONBACKEND_RequestInputByIndex(request, 0 /* index */, &input),
  //         "failed getting request input");
  //     if (input != nullptr) {
  //       const int64_t* shape = nullptr;
  //       LOG_IF_ERROR(
  //           TRITONBACKEND_InputProperties(
  //               input, nullptr, nullptr, &shape, nullptr, nullptr, nullptr),
  //           "failed getting input properties");
  //       if (shape != nullptr) {
  //         total_batch_size += shape[0];
  //       }
  //     }
  //   }
  // }
#else
  (void)exec_start_ns;
  (void)exec_end_ns;
  (void)receiver_start_ns;
  (void)receiver_end_ns;
  (void)responder_start_ns;
  (void)responder_end_ns;
  (void)compute_start_ns;
  (void)compute_end_ns;
#endif  // TRITON_ENABLE_STATS

  // Report statistics for each request, and then release the request.
  for (uint32_t r = 0; r < request_count; ++r) {
    auto& request = requests[r];

#ifdef TRITON_ENABLE_STATS
    LOG_IF_ERROR(
        TRITONBACKEND_ModelInstanceReportStatistics(
            this->TritonModelInstance(), request,
            (responses[r] != nullptr) /* success */, exec_start_ns,
            compute_start_ns, compute_end_ns, exec_end_ns),
        "failed reporting request statistics");
#endif  // TRITON_ENABLE_STATS

    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed releasing request");
  }
#ifdef TRITON_ENABLE_STATS
  LOG_MESSAGE(TRITONSERVER_LOG_INFO, "[fetch data from client => model inference => response to client]");
  uint64_t receiver_duration = (receiver_end_ns - receiver_start_ns)/1000;
  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
        ("ixrt_inference_receiver_duration_us " + std::to_string(receiver_duration)).c_str());

  uint64_t compute_infer_duration = (compute_end_ns - compute_start_ns)/1000;
  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
        ("ixrt_inference_compute_infer_duration_us " + std::to_string(compute_infer_duration)).c_str());

  uint64_t responder_duration = (responder_end_ns - responder_start_ns)/1000;
  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
        ("ixrt_inference_responder_duration_us " + std::to_string(responder_duration)).c_str());

  uint64_t request_duration = (exec_end_ns - exec_start_ns)/1000;
  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
        ("ixrt_inference_total_request_duration_us " + std::to_string(request_duration)).c_str());

  // Report batch statistics.
  LOG_IF_ERROR(
      TRITONBACKEND_ModelInstanceReportBatchStatistics(
          this->TritonModelInstance(), total_batch_size,
          exec_start_ns, compute_start_ns, compute_end_ns, exec_end_ns),
      "failed reporting batch request statistics");
#endif  // TRITON_ENABLE_STATS
  
}

}}}
