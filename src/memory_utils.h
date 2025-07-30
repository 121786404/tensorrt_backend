// Created on 2023/5/8.
#pragma once
#include <memory>
#include "NvInfer.h"
#include "cuda_runtime.h"

#ifndef CHECK
#define CHECK(status)                                          \
    do {                                                       \
        auto ret = (status);                                   \
        if (ret != 0) {                                        \
            std::cerr << "Cuda failure: " << ret << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)
#endif

struct ObjectDeleter {
    template <typename T>
    void operator()(T* obj) const {
        delete obj;
    }
};

struct ArrayDeleter {
    template <typename T>
    void operator()(T* p) const {
        delete[] p;
    }
};
template <typename T>
using UniquePtr = std::unique_ptr<T, ObjectDeleter>;

template <typename T>
using UniArrPtr = std::unique_ptr<T, ArrayDeleter>;

static auto StreamDeleter = [](cudaStream_t* pStream) {
    if (pStream) {
        cudaStreamDestroy(*pStream);
        delete pStream;
    }
};

inline std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> makeCudaStream() {
    std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> pStream(new cudaStream_t, StreamDeleter);
    if (cudaStreamCreateWithFlags(pStream.get(), cudaStreamNonBlocking) != cudaSuccess) {
        pStream.reset(nullptr);
    }

    return pStream;
}

static auto EventDeleter = [](cudaEvent_t* p_event) {
    if (p_event) {
        cudaEventDestroy(*p_event);
        delete p_event;
    }
};

inline std::unique_ptr<cudaEvent_t, decltype(EventDeleter)> makeCudaEvent(bool timing = false) {
    std::unique_ptr<cudaEvent_t, decltype(EventDeleter)> p_event(new cudaEvent_t, EventDeleter);
    if (timing) {
        if (cudaEventCreate(p_event.get()) != cudaSuccess) {
            p_event.reset(nullptr);
        }
    } else {
        if (cudaEventCreateWithFlags(p_event.get(), cudaEventDisableTiming) != cudaSuccess) {
            p_event.reset(nullptr);
        }
    }
    return p_event;
}

inline uint32_t getElementSize(nvinfer1::DataType t) noexcept {
    switch (t) {
        case nvinfer1::DataType::kINT64:
            return 8;
        case nvinfer1::DataType::kINT32:
        case nvinfer1::DataType::kFLOAT:
            return 4;
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kBOOL:
        case nvinfer1::DataType::kINT8:
        case nvinfer1::DataType::kUINT8:
            return 1;
    }
    return 0;
}

inline uint64_t GetBytes(const nvinfer1::Dims& dims, nvinfer1::DataType t) noexcept {
    uint64_t ret{1};
    for (auto i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] > 0) {
            ret *= dims.d[i];
        } else{
            std::cerr<< "Dim contains dynamic shape" << std::endl;
        }
    }
    if (ret >= 1) {
        return ret * getElementSize(t);
    } else {
        return 0;
    }
}