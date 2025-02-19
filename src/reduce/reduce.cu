#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include "qttbench/qtt_state.h"
#include "qttbench/qtt_tensor.cuh"

#define ERR_ 1e-3
__global__ void verify_gpu(const float *c, const float *a, int *ret)
{
    if (!(*ret))
        return;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // printf("%f %f\n", c[idx], a[idx]);
    if ((*ret) &&
        // This is not a good sanity check method, but in this experiment this is good enough.
        // refactor it with reduce sum mean diff
        (fabs(c[idx] - a[idx]) > ERR_ && fabs(c[idx] - a[idx]) / fabs(c[idx]) > ERR_))
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
        dim3 grid_size((num_heads+16-1)/16);                                                                    \
        dim3 block_size(16);                                                                      \
        int *d_correct;                                                                                 \
        int correct = 1;                                                                                \
        cudaMalloc(&d_correct, sizeof(int));                                                            \
        cudaMemcpy(d_correct, &correct, sizeof(int), cudaMemcpyHostToDevice);                           \
        verify_gpu<<<grid_size, block_size, 0, s>>>(output.data_ptr(), baseline.data_ptr(), d_correct); \
        cudaMemcpy(&correct, d_correct, sizeof(int), cudaMemcpyDeviceToHost);                           \
        reset_gpu<T><<<grid_size, block_size, 0, s>>>(output.data_ptr());                               \
        reset_gpu<T><<<grid_size, block_size, 0, s>>>(internal.data_ptr());                               \
        return correct;                                                                                 \
    }

template <typename T, int HD>
__global__ void reduce_v0_native(const T *input, T *output)
{
    __shared__ float sums[HD];

    int head_dim = blockDim.x * gridDim.x;
    int head_blk_x = blockIdx.x * blockDim.x;

    int idx = blockIdx.y*blockDim.y*head_dim + head_blk_x + threadIdx.x;
    int tid = threadIdx.x;
    sums[tid] = input[idx];
    __syncthreads();

    for(int s = 1; s<blockDim.x;s<<=1){
        if(tid % (2*s) == 0){ // mod operation is expensive
            sums[tid] += sums[tid+s];
        }
        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.y*gridDim.x+ blockIdx.x] = sums[0];
}

template <typename T, int HD>
__global__ void reduce_v1_interleaved_addressing(const T *input, T *output)
{
    __shared__ float sums[HD];
    int head_dim = blockDim.x * gridDim.x;
    int head_blk_x = blockIdx.x * blockDim.x;
    int idx = blockIdx.y*blockDim.y*head_dim + head_blk_x + threadIdx.x;
    int tid = threadIdx.x;
    sums[tid] = input[idx];
    __syncthreads();

    for(int s = 1; s<blockDim.x; s<<=1){
        int idx = 2 * s * tid;
        if(idx<blockDim.x){
            sums[idx] += sums[idx+s]; // share local memory bank conflict
        }
        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.y*gridDim.x+blockIdx.x] = sums[0];
}

template <typename T, int HD>
__global__ void reduce_v2_sequential_addressing(const T *input, T *output)
{
    __shared__ float sums[HD];
    int head_dim = blockDim.x * gridDim.x;
    int head_blk_x = blockIdx.x * blockDim.x;
    int idx = blockIdx.y*blockDim.y*head_dim + head_blk_x + threadIdx.x;
    int tid = threadIdx.x;
    sums[tid] = input[idx];
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s) // half of the threads are idle at the first loop
            sums[tid] += sums[s + tid];
        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.y*gridDim.x+blockIdx.x] = sums[0];
}

template <typename T, int HD>
__global__ void reduce_v3_first_add_during_load(const T *input, T *output, int head_dim)
{
    // for loop 0
    // blockDim.x=256
    // blockDim.y=1
    // gridDim.x=128
    // gridDim.y=16
    // HD=256

    // for loop 1
    // blockDim.x=64
    // blockDim.y=1
    // gridDim.x=1
    // gridDim.y=16
    // HD=64
    __shared__ float sums[HD];
    // int head_dim = blockDim.x * (gridDim.x<<1);
    // int head_blk_x = blockIdx.x * blockDim.x;
    int idx = blockIdx.y*blockDim.y*head_dim 
            + blockIdx.x * (blockDim.x <<1)
            + threadIdx.x;
    int tid = threadIdx.x;
    sums[tid] = input[idx]+input[idx+blockDim.x];
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if (tid < s)
            sums[tid] += sums[s + tid];
        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.y*gridDim.x+blockIdx.x] = sums[0];
}

template <typename T>
void test_with_dtype(qttbench::State &state)
{
    constexpr int head_dim = 256 * 256;
    constexpr int num_heads = 16;
    constexpr int BLK_SIZE = 256;

    // std::vector<int> ne = {head_dim, num_heads};

    qttbench::Tensor<qttbench::float32_t> input(2, {head_dim, num_heads});
    input.initialize_random();

    qttbench::Tensor<qttbench::float32_t> output(1, {num_heads});
    qttbench::Tensor<qttbench::float32_t> internal(2, {head_dim/BLK_SIZE, num_heads});

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
            dim3 grid_size(head_dim/BLK_SIZE, num_heads);
            dim3 block_size(BLK_SIZE, 1);
            // 256*256 -> 256
            reduce_v0_native<T, BLK_SIZE><<<grid_size, block_size, 0, s>>>(
                static_cast<const T *>(input.data_ptr()), internal.data_ptr());
            // 256 -> 1
            dim3 grid_size_1(head_dim/BLK_SIZE/BLK_SIZE, num_heads);
            dim3 block_size_1(BLK_SIZE, 1);
            reduce_v0_native<T, BLK_SIZE><<<grid_size_1, block_size_1, 0, s>>>(
                static_cast<const T *>(internal.data_ptr()), output.data_ptr());
        },
        VERIFY_FUNC);

    state.run(
        "reduce_v1_interleaved_addressing",
        [&](cudaStream_t s)
        {
            dim3 grid_size(head_dim/BLK_SIZE, num_heads);
            dim3 block_size(BLK_SIZE, 1);
            // 256*256 -> 256
            reduce_v1_interleaved_addressing<T, BLK_SIZE><<<grid_size, block_size, 0, s>>>(
                static_cast<const T *>(input.data_ptr()), internal.data_ptr());
            // 256 -> 1
            dim3 grid_size_1(head_dim/BLK_SIZE/BLK_SIZE, num_heads);
            dim3 block_size_1(BLK_SIZE, 1);
            reduce_v1_interleaved_addressing<T, BLK_SIZE><<<grid_size_1, block_size_1, 0, s>>>(
                static_cast<const T *>(internal.data_ptr()), output.data_ptr());
        },
        VERIFY_FUNC);

    state.run(
        "reduce_v2_sequential_addressing",
        [&](cudaStream_t s)
        {
            dim3 grid_size(head_dim/BLK_SIZE, num_heads);
            dim3 block_size(BLK_SIZE, 1);
            // 256*256 -> 256
            reduce_v2_sequential_addressing<T, BLK_SIZE><<<grid_size, block_size, 0, s>>>(
                static_cast<const T *>(input.data_ptr()), internal.data_ptr());
            // 256 -> 1
            dim3 grid_size_1(head_dim/BLK_SIZE/BLK_SIZE, num_heads);
            dim3 block_size_1(BLK_SIZE, 1);
            reduce_v2_sequential_addressing<T, BLK_SIZE><<<grid_size_1, block_size_1, 0, s>>>(
                static_cast<const T *>(internal.data_ptr()), output.data_ptr());
        },
        VERIFY_FUNC);

        state.run(
        "reduce_v3_first_add_during_load",
        [&](cudaStream_t s)
        {
            int N_BLK=(head_dim+BLK_SIZE-1)/BLK_SIZE/2;
            dim3 grid_size(N_BLK, num_heads);
            dim3 block_size(BLK_SIZE, 1);
            // 256*256 -> 128
            reduce_v3_first_add_during_load<T, BLK_SIZE><<<grid_size, block_size, 0, s>>>(
                static_cast<const T *>(input.data_ptr()), internal.data_ptr(), head_dim);
            // 128 -> 1
            constexpr int blk_size_1=BLK_SIZE/4;
            dim3 grid_size_1((N_BLK+blk_size_1-1)/blk_size_1/2, num_heads);
            dim3 block_size_1(blk_size_1, 1);
            // printf("grid_size_1 %d %d %d\n", grid_size_1.x, grid_size_1.y, grid_size_1.z);
            
            reduce_v3_first_add_during_load<T, blk_size_1><<<grid_size_1, block_size_1, 0, s>>>(
                static_cast<const T *>(internal.data_ptr()), output.data_ptr(), head_dim/BLK_SIZE/2);
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
