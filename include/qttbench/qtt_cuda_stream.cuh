#ifndef QTT_CUDASTREAM
#define QTT_CUDASTREAM

#include "qtt_cuda_runtime.cuh"
#include "qtt_cuda_call.cuh"
#include "utils.h"

namespace qttbench
{
    struct cuda_stream
    {
        cuda_stream() : qtt_cuda_stream(
                            []()
                            {
                                cudaStream_t cst;
                                QTTBENCH_CUDA_CALL(cudaStreamCreate(&cst));
                                return cst;
                            }(),
                            stream_deleter()) {};
        __forceinline__ cudaStream_t get()
        {
            cudaStream_t cst = qtt_cuda_stream.get();
            if (cst)
            {
                return qtt_cuda_stream.get();
            }
            else
            {
                throw std::runtime_error(strutils::string_format("%s:%d: cuda_stream (cudastream_t*)get() method returns a nullptr.\n"));
            }
        };

    private:
        struct stream_deleter
        {
            using pointer = cudaStream_t;

            void operator()(pointer s) const noexcept
            {
                QTTBENCH_CUDA_CALL_NOEXCEPT(cudaStreamDestroy(s));
            }
        };
        std::unique_ptr<cudaStream_t, stream_deleter> qtt_cuda_stream;
    };

}

#endif