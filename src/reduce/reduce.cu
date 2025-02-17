#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include "qttbench/qtt_state.h"
#include "qttbench/qtt_tensor.cuh"

__global__ void verify_gpu(const float *c, const float *a, int *ret)
{
    if (!(*ret))
        return;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if ((*ret) &&
        // This is not a good sanity check method, but in this experiment this is good enough.
        // refactor it with reduce sum mean diff
        fabs(c[idx] - a[idx]) > 1e-3)
    {
        printf("%f %f\n", c[idx], a[idx]);
        (*ret) = 0;
    }
}

template <typename T>
__global__ void reset_gpu(T *m_t)
{
    m_t[blockIdx.x * blockDim.x + threadIdx.x] = 0.f;
}

#define VERIFY_FUNC                                                                                     \
    [&](cudaStream_t s)                                                                                 \
    {                                                                                                   \
        dim3 grid_size(num_heads/16);                                                                    \
        dim3 block_size(16);                                                                      \
        int *d_correct;                                                                                 \
        int correct = 1;                                                                                \
        cudaMalloc(&d_correct, sizeof(int));                                                            \
        cudaMemcpy(d_correct, &correct, sizeof(int), cudaMemcpyHostToDevice);                           \
        verify_gpu<<<grid_size, block_size, 0, s>>>(output.data_ptr(), baseline.data_ptr(), d_correct); \
        cudaMemcpy(&correct, d_correct, sizeof(int), cudaMemcpyDeviceToHost);                           \
        reset_gpu<T><<<grid_size, block_size, 0, s>>>(output.data_ptr());                               \
        return correct;                                                                                 \
    }

template <typename T, int HD>
__global__ void reduce_native(const T *input, T *output)
{
    __shared__ float sums[HD];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    sums[tid] = input[idx];
    __syncthreads();

    for (int i = HD / 2; i > 0; i >>= 1)
    {
        if (tid < i)
            sums[tid] += sums[i + tid];
        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = sums[0];
}

template <typename T>
void test_with_dtype(qttbench::State &state)
{
    constexpr int head_dim = 256;
    constexpr int num_heads = 32 * 1024;

    // std::vector<int> ne = {head_dim, num_heads};

    qttbench::Tensor<qttbench::float32_t> input(2, {head_dim, num_heads});
    input.initialize_random();

    qttbench::Tensor<qttbench::float32_t> output(1, {num_heads});

    T sums[num_heads] = {0};
    T* i_data_cpu = (T*)malloc(sizeof(T)*head_dim*num_heads);
    input.cpu_data(i_data_cpu);
    for (int i = 0; i < num_heads; i++)
    {   
        for (int j = 0; j < head_dim; j++)
        {
            sums[i] += i_data_cpu[head_dim*i+j];
        } 
    }

    qttbench::Tensor<qttbench::float32_t> baseline(1, {num_heads}, sums);

    state.run(
        "reduce_native",
        [&](cudaStream_t s)
        {
            dim3 grid_size(num_heads);
            dim3 block_size(head_dim);
            reduce_native<T, head_dim><<<grid_size, block_size, 0, s>>>(
                static_cast<const T *>(input.data_ptr()), output.data_ptr());
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
