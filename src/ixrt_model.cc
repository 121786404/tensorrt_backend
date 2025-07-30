#include "ixrt_model.h"
#include "triton/backend/backend_common.h"
#include "triton/common/triton_json.h"
#include "triton/common/logging.h"

namespace triton { namespace backend { namespace ixrt {

TRITONSERVER_Error*
ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state)
{

  LOG_VERBOSE(2) << "ModelState::Create()";

  try {
    *state = new ModelState(triton_model);
  }
  catch (const BackendModelException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelException"));
    RETURN_IF_ERROR(ex.err_);
  }
  
  // step1: get model repo path
  const char* location;
  TRITONBACKEND_ArtifactType type;

  TRITONBACKEND_ModelRepository(triton_model, &type, &location);
  std::string model_repository_path(location);

  // step2: get model version
  uint64_t version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(triton_model, &version));

  // step3: get pre-built model engine according to model path + version
  // we will get the key-values likes {model_name - model_path} by calling ModelPaths().
  std::unordered_map<std::string, std::string> model_paths;
  ModelPaths(model_repository_path, version, false, false, &model_paths);

  // step4: get model configuration, and convert it to json style.
  TRITONSERVER_Message *model_config_message;
  RETURN_IF_ERROR(TRITONBACKEND_ModelConfig(triton_model, version, &model_config_message));

  const char* buffer;
  size_t byte_size;
  RETURN_IF_ERROR(TRITONSERVER_MessageSerializeToJson(
      model_config_message, &buffer, &byte_size));
  // model config likes this: {"name":"resnet50","platform":"","backend":"ixrt","version_policy":{"latest":{"num_versions":1}},"max_batch_size":16,"input":[{"name":"input","data_type":"TYPE_FP32","format":"FORMAT_NONE","dims":[3,224,224],"is_shape_tensor":false,"allow_ragged_batch":false,"optional":false}],"output":[{"name":"output","data_type":"TYPE_FP32","dims":[1000],"label_filename":"","is_shape_tensor":false}],"batch_input":[],"batch_output":[],"optimization":{"priority":"PRIORITY_DEFAULT","input_pinned_memory":{"enable":true},"output_pinned_memory":{"enable":true},"gather_kernel_buffer_threshold":0,"eager_batching":false},"instance_group":[{"name":"resnet50_0","kind":"KIND_CPU","count":1,"gpus":[],"secondary_devices":[],"profile":[],"passive":false,"host_policy":""}],"default_model_filename":"","cc_model_filenames":{},"metric_tags":{},"parameters":{},"model_warmup":[]}
  
  if (byte_size != 0) {
    (*state)->config_.Parse(buffer, byte_size);
  } 

  if (model_paths.size() > 1)
  {
    LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "!!! Only support 1 engine Per Model now!");
  }

  // step5: update loaded_models key-value map, and load pre-built engine.
  std::string model_path;
  for (auto iter = model_paths.begin(); iter != model_paths.end(); ++iter) {
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("TRITONBACKEND_ModelInitialize: scanning ") + iter->second).c_str());
    
    // update model file name
    (*state)->config_.SetStringObject("default_model_filename", iter->first);

    // more version could add to "default_model_filename"
    // ...
  }

  return nullptr;  // success
}

}}}  // namespace triton::backend::ixrt