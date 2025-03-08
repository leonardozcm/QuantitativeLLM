#ifndef QTT_CUDA_CALL
#define QTT_CUDA_CALL

#include <cuda_runtime_api.h>
#include "utils.h"

/// Throws a std::runtime_error if `call` doesn't return `cudaSuccess`.
/// Resets the error with cudaGetLastError().
#define QTTBENCH_CUDA_CALL(call)                                                                    \
  do                                                                                               \
  {                                                                                                \
    const cudaError_t qtt_cuda_call_error = call;                                              \
    if (qtt_cuda_call_error != cudaSuccess)                                                    \
    {                                                                                              \
      cudaGetLastError();                                                                          \
      strutils::throw_error(__FILE__, __LINE__, #call, qtt_cuda_call_error);         \
    }                                                                                              \
  } while (false);

/// Terminates process with failure status if `call` doesn't return
/// `cudaSuccess`.
#define QTTBENCH_CUDA_CALL_NOEXCEPT(call)                                                           \
  do                                                                                               \
  {                                                                                                \
    const cudaError_t qtt_cuda_call_error = call;                                              \
    if (qtt_cuda_call_error != cudaSuccess)                                                    \
    {                                                                                              \
      strutils::exit_error(__FILE__, __LINE__, #call, qtt_cuda_call_error);          \
    }                                                                                              \
  } while (false);

#endif
