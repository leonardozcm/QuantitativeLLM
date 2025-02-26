#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include "qttbench/qtt_state.h"
#include "qttbench/qtt_tensor.cuh"

__global__ void verify_gpu(const float *output, const float *baseline, int *ret)
{
    if (!(*ret))
        return;

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // printf("%f, %f\n", c[idx], a[idx]);
    // race but matters not
    if ((*ret) &&
        // This is not a good sanity check method, but in this experiment this is good enough.
        // refactor it with reduce sum mean diff
        fabs(baseline[idx]) > 0.001 &&
        fabs((output[idx] - baseline[idx]) / fmax(baseline[idx], output[idx])) > 0.02)
    {
        printf("%f %f\n", output[idx], baseline[idx]);
        (*ret) = 0;
    }
}

template <typename T>
__global__ void reset_gpu(T *m_t, const int HD = 128)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    m_t[idx] = 0.f;
}

#define VERIFY_FUNC                                                                                     \
    [&](cudaStream_t s)                                                                                 \
    {                                                                                                   \
        constexpr int block_width = 256;                                                                \
        const size_t n_elements = baseline.n_elements();                                                \
        dim3 grid_size(n_elements / block_width, 1, 1);                                                 \
        dim3 block_size(block_width, 1, 1);                                                             \
        int *d_correct;                                                                                 \
        int correct = 1;                                                                                \
        cudaMalloc(&d_correct, sizeof(int));                                                            \
        cudaMemcpy(d_correct, &correct, sizeof(int), cudaMemcpyHostToDevice);                           \
        verify_gpu<<<grid_size, block_size, 0, s>>>(output.data_ptr(), baseline.data_ptr(), d_correct); \
        cudaMemcpy(&correct, d_correct, sizeof(int), cudaMemcpyDeviceToHost);                           \
        reset_gpu<T><<<grid_size, block_size, 0, s>>>(output.data_ptr());                               \
        return correct;                                                                                 \
    }

template <int blockSize>
__device__ void warpReduceMax(volatile float *sdata, int tid)
{
    if (blockSize >= 64)
        sdata[tid] = max(sdata[tid], sdata[tid + 32]);
    if (blockSize >= 32)
        sdata[tid] = max(sdata[tid], sdata[tid + 16]);
    if (blockSize >= 16)
        sdata[tid] = max(sdata[tid], sdata[tid + 8]);
    if (blockSize >= 8)
        sdata[tid] = max(sdata[tid], sdata[tid + 4]);
    ;
    if (blockSize >= 4)
        sdata[tid] = max(sdata[tid], sdata[tid + 2]);
    ;
    if (blockSize >= 2)
        sdata[tid] = max(sdata[tid], sdata[tid + 1]);
    ;
}

template <int blockSize>
__device__ void warpReduceSum(volatile float *sdata, int tid)
{
    if (blockSize >= 64)
        sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32)
        sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16)
        sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8)
        sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4)
        sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2)
        sdata[tid] += sdata[tid + 1];
}

template <typename T, int BLOCK_SIZE = 256> // == blk_size
__global__ void rms_native(const T *input, T *output, const int length)
{
    int seq_offset = blockIdx.x * length;
    float *input_ptr = (float *)input + seq_offset;
    float *output_ptr = (float *)output + seq_offset;

    int tid = threadIdx.x;
    __shared__ float maxreduce_data[BLOCK_SIZE];

    // Find the maximum value in the block
    float max_tid = 0;
    for (int i = tid; i < length; i += blockDim.x)
    {
        max_tid = max_tid;
    }
    maxreduce_data[tid] = max_tid;
    __syncthreads();

    if (BLOCK_SIZE >= 256)
    {
        if (tid < 128)
        {
            maxreduce_data[tid] = max(maxreduce_data[tid], maxreduce_data[tid + 128]);
        }
        __syncthreads();
    }

    if (BLOCK_SIZE >= 128)
    {
        if (tid < 64)
        {
            maxreduce_data[tid] = max(maxreduce_data[tid], maxreduce_data[tid + 64]);
        }
        __syncthreads();
    }

    if (tid < 32)
    {
        warpReduceMax<BLOCK_SIZE>(maxreduce_data, tid);
    }
    __syncthreads();

    float max_val = maxreduce_data[0];
    float exp_sum = 0.f;

    for (int i = tid; i < length; i += blockDim.x)
    {
        float val = expf(input_ptr[i] - max_val);
        output_ptr[i] = val;
        exp_sum += val;
    }
    maxreduce_data[tid] = exp_sum;
    __syncthreads();

    if (BLOCK_SIZE >= 256)
    {
        if (tid < 128)
        {
            maxreduce_data[tid] = maxreduce_data[tid] + maxreduce_data[tid + 128];
        }
        __syncthreads();
    }

    if (BLOCK_SIZE >= 128)
    {
        if (tid < 64)
        {
            maxreduce_data[tid] = maxreduce_data[tid] + maxreduce_data[tid + 64];
        }
        __syncthreads();
    }

    if (tid < 32)
    {
        warpReduceSum<BLOCK_SIZE>(maxreduce_data, tid);
    }
    __syncthreads();

    exp_sum = maxreduce_data[0];

    for (int i = tid; i < length; i += blockDim.x)
    {
        output_ptr[i] /= exp_sum; // most of them are cached in L2
    }
}

template <typename T>
void test_with_dtype(qttbench::State &state)
{
    constexpr int bs = 1, seq_len = 1024, head_embed = 4096; // 32 * 128

    std::vector<int> ne = {head_embed, seq_len, bs};

    qttbench::Tensor<qttbench::float32_t> input(3, ne);
    input.initialize_random();

    qttbench::Tensor<qttbench::float32_t> baseline(3, ne);
    qttbench::Tensor<qttbench::float32_t> output(3, ne);

    state.run(
        "rms_norm_native",
        [&](cudaStream_t s)
        {
            constexpr int block_width = 256;
            dim3 grid_size(seq_len);
            dim3 block_size(block_width);
            rms_native<T, block_width><<<grid_size, block_size, 0, s>>>(
                static_cast<const T *>(input.data_ptr()), output.data_ptr(), context_len);
        },
        VERIFY_FUNC);
}

int main(int argc, char *argv[])
{
    int turns = 1;
    int perf = 1;
    char *turns_t = strutils::getCmdOption(argv, argv + argc, "-t");
    char *perf_t = strutils::getCmdOption(argv, argv + argc, "-p");
    if (turns_t)
    {
        try
        {
            turns = std::stoi(turns_t);
        }
        catch (const std::invalid_argument &e)
        {
            std::cerr << "Invalid number " << turns_t << std::endl;
        }
        catch (const std::out_of_range &e)
        {
            std::cerr << "Number out of range: " << turns_t << '\n';
        }
    }

    if (perf_t)
    {
        try
        {
            perf = std::stoi(perf_t) == 0;
        }
        catch (const std::invalid_argument &e)
        {
            std::cerr << "Invalid number " << perf_t << std::endl;
        }
        catch (const std::out_of_range &e)
        {
            std::cerr << "Number out of range: " << perf_t << '\n';
        }
    }

    qttbench::State state(turns, perf, perf);
    state.set_csv_output(strutils::get_filename_without_extension(__FILE__));

    test_with_dtype<qttbench::float32_t>(state);
    return 0;
}
