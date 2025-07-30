#pragma once
#include "triton/backend/backend_model.h"

namespace triton { namespace backend { namespace ixrt {

//
// ModelState
//
// State associated with a model that is using this backend. An object
// of this class is created and associated with each
// TRITONBACKEND_Model. ModelState is derived from BackendModel class
// provided in the backend utilities that provides many common
// functions.
//
class ModelState : public BackendModel {
 public:
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Model* triton_model, ModelState** state);
  
  ~ModelState() {
    // release key-value resources.
    config_.Release();
  };

 private:
  ModelState(TRITONBACKEND_Model* triton_model) : BackendModel(triton_model) {}

 public:
  triton::common::TritonJson::Value config_;
};

}}}  // namespace triton::backend::ixrt