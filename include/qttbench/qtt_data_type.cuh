#pragma once
#include "qtt_cuda_runtime.cuh"

// todo: employing a micro to define the cuda data type name space
namespace qttbench{
    using int8_t    = int8_t;
    using int16_t   = int16_t;
    using int32_t   = int32_t;
    using int64_t   = int64_t;
    using uint8_t   = uint8_t;
    using uint16_t  = uint16_t;
    using uint32_t  = uint32_t;
    using uint64_t  = uint64_t;
    using float32_t = float;
    using float64_t = double;
}
