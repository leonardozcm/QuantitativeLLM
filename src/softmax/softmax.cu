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

#define VERIFY_FUNC                                                                                              \
    [&](cudaStream_t s)                                                                                          \
    {                                                                                                            \
        constexpr int block_width = 256;                                                                          \
        const size_t n_elements = baseline.n_elements(); \
        dim3 grid_size(n_elements / block_width,1,1);                          \
        dim3 block_size(block_width, 1, 1);                                                           \
        int *d_correct;                                                                                          \
        int correct = 1;                                                                                         \
        cudaMalloc(&d_correct, sizeof(int));                                                                     \
        cudaMemcpy(d_correct, &correct, sizeof(int), cudaMemcpyHostToDevice);                                    \
        verify_gpu<<<grid_size, block_size, 0, s>>>(output.data_ptr(), baseline.data_ptr(), d_correct); \
        cudaMemcpy(&correct, d_correct, sizeof(int), cudaMemcpyDeviceToHost);                                    \
        reset_gpu<T><<<grid_size, block_size, 0, s>>>(output.data_ptr());                               \
        return correct;                                                                                          \
    }

// Input Tensor: [bs, n_heads, seq_length, context_length]
// Output Tensor: [bs, n_heads, seq_length, context_length]
//
// let i belongs to [0, context_length)
// output[bs, n_heads, seq_length, i] = e ^ input[bs, n_heads, seq_length, i] / (sum of (e ^ input[bs, n_heads, seq_length, j]) for j in [0, context_length))

#define WARP_SIZE 32

template<int width = WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_max(float x) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, offset, width));
    }
    return x;
}

static __device__ __forceinline__ float warp_reduce_sum(float x) {
    #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            x += __shfl_xor_sync(0xffffffff, x, offset, 32);
        }
        return x;
    }

template <bool use_shared, int ncols_template, int block_size_template>
static __global__ void soft_max_f32(
        const float * x, float * dst,
        const int ncols_par, // context_length
        const int nrows_y, // rows
        const float scale, // 1.0f by default
        uint32_t n_head_log2) {
    const int ncols = ncols_template == 0 ? ncols_par : ncols_template;

    const int tid  = threadIdx.x;
    const int rowx = blockIdx.x;
    // const int rowy = rowx % nrows_y; // broadcast the mask in the row dimension

    x    += int64_t(rowx)*ncols;
    // mask += int64_t(rowy)*ncols * (mask != nullptr);
    dst  += int64_t(rowx)*ncols;

    const int block_size = block_size_template == 0 ? blockDim.x : block_size_template;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    // unused in this case
    // const float slope = get_alibi_slope(max_bias, rowx/nrows_y, n_head_log2, m0, m1);

    extern __shared__ float data_soft_max_f32[];
    float * buf_iw = data_soft_max_f32; // shared memory buffer for inter-warp communication
    // shared memory buffer to cache values between iterations:
    float * vals = use_shared ? buf_iw + WARP_SIZE : dst;

    float max_val = -INFINITY;

#pragma unroll
    for (int col0 = 0; col0 < ncols; col0 += block_size) {
        const int col = col0 + tid;

        if (ncols_template == 0 && col >= ncols) {
            break;
        }

        // const float val = x[col]*scale + (mask ? slope*t2f32(mask[col]) : 0.0f);
        const float val = x[col]*scale;

        vals[col] = val;
        max_val = max(max_val, val);
    }

    // find the max value in the block
    max_val = warp_reduce_max(max_val);
    if (block_size > WARP_SIZE) {
        if (warp_id == 0) {
            buf_iw[lane_id] = -INFINITY;
        }
        __syncthreads();

        if (lane_id == 0) {
            buf_iw[warp_id] = max_val;
        }
        __syncthreads();

        max_val = buf_iw[lane_id];
        max_val = warp_reduce_max(max_val);
    }

    float tmp = 0.0f; // partial sum

#pragma unroll
    for (int col0 = 0; col0 < ncols; col0 += block_size) {
        const int col = col0 + tid;

        if (ncols_template == 0 && col >= ncols) {
            break;
        }

        const float val = expf(vals[col] - max_val);
        tmp += val;
        vals[col] = val;
    }

    // find the sum of exps in the block
    tmp = warp_reduce_sum(tmp);
    if (block_size > WARP_SIZE) {
        __syncthreads();
        if (warp_id == 0) {
            buf_iw[lane_id] = 0.0f;
        }
        __syncthreads();

        if (lane_id == 0) {
            buf_iw[warp_id] = tmp;
        }
        __syncthreads();

        tmp = buf_iw[lane_id];
        tmp = warp_reduce_sum(tmp);
    }

    const float inv_sum = 1.0f / tmp;

#pragma unroll
    for (int col0 = 0; col0 < ncols; col0 += block_size) {
        const int col = col0 + tid;

        if (ncols_template == 0 && col >= ncols) {
            return;
        }

        dst[col] = vals[col] * inv_sum;
    }
}

template <int blockSize>
__device__ void warpReduceMax(volatile float* sdata, int tid){
    if(blockSize >= 64) sdata[tid] = max(sdata[tid], sdata[tid + 32]);
    if(blockSize >= 32) sdata[tid] = max(sdata[tid], sdata[tid + 16]);
    if(blockSize >= 16) sdata[tid] = max(sdata[tid], sdata[tid + 8]);
    if(blockSize >= 8) sdata[tid] =  max(sdata[tid], sdata[tid + 4]);;
    if(blockSize >= 4) sdata[tid] =  max(sdata[tid], sdata[tid + 2]);;
    if(blockSize >= 2) sdata[tid] =  max(sdata[tid], sdata[tid + 1]);;
}

template <int blockSize>
__device__ void warpReduceSum(volatile float* sdata, int tid){
    if(blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if(blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if(blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if(blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if(blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if(blockSize >= 2) sdata[tid] += sdata[tid + 1];
}


template <typename T, int BLOCK_SIZE=256> // == blk_size
__global__ void softmax_native(const T *input, T *output, const int context_length)
{
    int seq_offset = blockIdx.x * context_length;
    float *input_ptr = (float *)input + seq_offset;
    float *output_ptr = (float *)output + seq_offset;

    int tid = threadIdx.x;
    __shared__ float maxreduce_data[BLOCK_SIZE];

    // Find the maximum value in the block
    float max_tid = -INFINITY;
    for(int i=tid; i<context_length; i+=blockDim.x)
    {
        max_tid = max(input_ptr[i], max_tid);
    }
    maxreduce_data[tid] = max_tid;
    __syncthreads();

    if(BLOCK_SIZE>=256){
        if(tid<128){
            maxreduce_data[tid] = max(maxreduce_data[tid], maxreduce_data[tid+128]);
        }
        __syncthreads();
    }

    if(BLOCK_SIZE>=128){
        if(tid<64){
            maxreduce_data[tid] = max(maxreduce_data[tid], maxreduce_data[tid+64]);
        }
        __syncthreads();
    } 

    if(tid<32){
        warpReduceMax<BLOCK_SIZE>(maxreduce_data, tid);
    }
    __syncthreads();

    float max_val = maxreduce_data[0];
    float exp_sum = 0.f;

    for(int i=tid; i<context_length; i+=blockDim.x)
    {
        float val = expf(input_ptr[i] - max_val);
        output_ptr[i] = val;
        exp_sum += val;
    }
    maxreduce_data[tid]=exp_sum;
    __syncthreads();

    if(BLOCK_SIZE>=256){
        if(tid<128){
            maxreduce_data[tid] = maxreduce_data[tid] + maxreduce_data[tid+128];
        }
        __syncthreads();
    }

    if(BLOCK_SIZE>=128){
        if(tid<64){
            maxreduce_data[tid] = maxreduce_data[tid] + maxreduce_data[tid+64];
        }
        __syncthreads();
    } 
    
    if(tid<32){
        warpReduceSum<BLOCK_SIZE>(maxreduce_data, tid);
    }
    __syncthreads();

    exp_sum = maxreduce_data[0];

    for(int i=tid; i<context_length; i+=blockDim.x)
    {
        output_ptr[i] /= exp_sum;
    }

}

#define GGML_PAD(x, n) (((x) + (n) - 1) & ~((n) - 1))

template <typename T>
void test_with_dtype(qttbench::State &state)
{
    constexpr int bs = 1, num_heads = 1, seq_len = 1024, context_len = 1024;

    std::vector<int> ne = {context_len, num_heads, seq_len, bs};

    qttbench::Tensor<qttbench::float32_t> input(4, ne);
    input.initialize_random();

    qttbench::Tensor<qttbench::float32_t> baseline(4, ne);
    qttbench::Tensor<qttbench::float32_t> output(4, ne);

    state.run(
        "softmax_ggml",
        [&](cudaStream_t s)
        {
            int nth=WARP_SIZE;
            const int CUDA_SOFT_MAX_BLOCK_SIZE=512;
            const int ncols_x = context_len;
            const int nrows_x = num_heads*seq_len;
            const int nrows_y = seq_len;
            while (nth < ncols_x && nth < CUDA_SOFT_MAX_BLOCK_SIZE) nth *= 2;
            const size_t shmem = (GGML_PAD(ncols_x, WARP_SIZE) + WARP_SIZE)*sizeof(float);

            const uint32_t n_head      = nrows_x/nrows_y;
            const uint32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head));

            dim3 block_nums(nrows_x, 1, 1);
            dim3 block_dims(nth, 1, 1);
            soft_max_f32<true, context_len, CUDA_SOFT_MAX_BLOCK_SIZE><<<block_nums, block_dims, shmem, s>>>(
                input.data_ptr(),
                baseline.data_ptr(),
                ncols_x, nrows_y, 1.0f, n_head_log2);
        },
        [&](cudaStream_t s)
        {
            return true;
        });

    state.run(
        "softmax_native",
        [&](cudaStream_t s)
        {
            constexpr int block_width = 256;
            dim3 grid_size(num_heads*seq_len);
            dim3 block_size(block_width);
            softmax_native<T, block_width><<<grid_size, block_size, 0, s>>>(
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
