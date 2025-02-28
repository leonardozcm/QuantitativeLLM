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


#define WARP_SIZE 32

static __device__ __forceinline__ float warp_reduce_sum(float x)
{
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        x += __shfl_xor_sync(0xffffffff, x, offset, 32);
    }
    return x;
}

template <int block_size>
static __global__ void rms_norm_f32(const float * x, float * dst, const int ncols, const float eps) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    float tmp = 0.0f; // partial sum for thread in warp

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[row*ncols + col];
        tmp += xi * xi;
    }

    // sum up partial sums
    tmp = warp_reduce_sum(tmp);
    if (block_size > WARP_SIZE) {
        __shared__ float s_sum[32];
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        __syncthreads();
        tmp = s_sum[lane_id];
        tmp = warp_reduce_sum(tmp);
    }

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        dst[row*ncols + col] = scale * x[row*ncols + col];
    }
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
        max_tid += input_ptr[i] * input_ptr[i];
    }
    maxreduce_data[tid] = max_tid;
    __syncthreads();

    if (BLOCK_SIZE >= 256)
    {
        if (tid < 128)
        {
            maxreduce_data[tid] += maxreduce_data[tid + 128];
        }
        __syncthreads();
    }

    if (BLOCK_SIZE >= 128)
    {
        if (tid < 64)
        {
            maxreduce_data[tid] += maxreduce_data[tid + 64];
        }
        __syncthreads();
    }

    if (tid < 32)
    {
        warpReduceSum<BLOCK_SIZE>(maxreduce_data, tid);
    }
    __syncthreads();

    float pow_sum = maxreduce_data[0];
    float rms_sum = sqrtf(pow_sum/length);

    for (int i = tid; i < length; i += blockDim.x)
    {
        output_ptr[i] = input_ptr[i] / rms_sum; // most of them are cached in L2
    }
}

template <typename T, int BLOCK_SIZE = 256, int PADDING_LENGTH=4096> // == blk_size
__global__ void rms_unroll(const T *input, T *output)
{
    int seq_offset = blockIdx.x * PADDING_LENGTH;
    float *input_ptr = (float *)input + seq_offset;
    float *output_ptr = (float *)output + seq_offset;

    int tid = threadIdx.x;
    __shared__ float maxreduce_data[BLOCK_SIZE];

    // Find the maximum value in the block
    float max_tid = 0;
    int idx=tid;
    {
        max_tid += input_ptr[idx] * input_ptr[idx];idx+=BLOCK_SIZE; 
        max_tid += input_ptr[idx] * input_ptr[idx];idx+=BLOCK_SIZE; 
        max_tid += input_ptr[idx] * input_ptr[idx];idx+=BLOCK_SIZE; 
        max_tid += input_ptr[idx] * input_ptr[idx];idx+=BLOCK_SIZE; 
    }
    {
        max_tid += input_ptr[idx] * input_ptr[idx];idx+=BLOCK_SIZE; 
        max_tid += input_ptr[idx] * input_ptr[idx];idx+=BLOCK_SIZE; 
        max_tid += input_ptr[idx] * input_ptr[idx];idx+=BLOCK_SIZE; 
        max_tid += input_ptr[idx] * input_ptr[idx];idx+=BLOCK_SIZE; 
    }
    {
        max_tid += input_ptr[idx] * input_ptr[idx];idx+=BLOCK_SIZE; 
        max_tid += input_ptr[idx] * input_ptr[idx];idx+=BLOCK_SIZE; 
        max_tid += input_ptr[idx] * input_ptr[idx];idx+=BLOCK_SIZE; 
        max_tid += input_ptr[idx] * input_ptr[idx];idx+=BLOCK_SIZE; 
    }
    {
        max_tid += input_ptr[idx] * input_ptr[idx];idx+=BLOCK_SIZE; 
        max_tid += input_ptr[idx] * input_ptr[idx];idx+=BLOCK_SIZE; 
        max_tid += input_ptr[idx] * input_ptr[idx];idx+=BLOCK_SIZE; 
        max_tid += input_ptr[idx] * input_ptr[idx];idx+=BLOCK_SIZE; 
    }

    maxreduce_data[tid] = max_tid;
    __syncthreads();

    if (BLOCK_SIZE >= 256)
    {
        if (tid < 128)
        {
            maxreduce_data[tid] += maxreduce_data[tid + 128];
        }
        __syncthreads();
    }

    if (BLOCK_SIZE >= 128)
    {
        if (tid < 64)
        {
            maxreduce_data[tid] += maxreduce_data[tid + 64];
        }
        __syncthreads();
    }

    if (tid < 32)
    {
        warpReduceSum<BLOCK_SIZE>(maxreduce_data, tid);
    }
    __syncthreads();

    float pow_sum = maxreduce_data[0];
    float rsqrt_rms_sum = rsqrtf(pow_sum/PADDING_LENGTH);// avoid divide instructions and rsqrt has faster inplementation, also this decrease the arithmetic intensity.
    

    idx=tid;
    {
        output_ptr[idx] = input_ptr[idx] * rsqrt_rms_sum;idx+=BLOCK_SIZE;
        output_ptr[idx] = input_ptr[idx] * rsqrt_rms_sum;idx+=BLOCK_SIZE;
        output_ptr[idx] = input_ptr[idx] * rsqrt_rms_sum;idx+=BLOCK_SIZE;
        output_ptr[idx] = input_ptr[idx] * rsqrt_rms_sum;idx+=BLOCK_SIZE;
    }
    {
        output_ptr[idx] = input_ptr[idx] * rsqrt_rms_sum;idx+=BLOCK_SIZE;
        output_ptr[idx] = input_ptr[idx] * rsqrt_rms_sum;idx+=BLOCK_SIZE;
        output_ptr[idx] = input_ptr[idx] * rsqrt_rms_sum;idx+=BLOCK_SIZE;
        output_ptr[idx] = input_ptr[idx] * rsqrt_rms_sum;idx+=BLOCK_SIZE;
    }
    {
        output_ptr[idx] = input_ptr[idx] * rsqrt_rms_sum;idx+=BLOCK_SIZE;
        output_ptr[idx] = input_ptr[idx] * rsqrt_rms_sum;idx+=BLOCK_SIZE;
        output_ptr[idx] = input_ptr[idx] * rsqrt_rms_sum;idx+=BLOCK_SIZE;
        output_ptr[idx] = input_ptr[idx] * rsqrt_rms_sum;idx+=BLOCK_SIZE;
    }
    {
        output_ptr[idx] = input_ptr[idx] * rsqrt_rms_sum;idx+=BLOCK_SIZE;
        output_ptr[idx] = input_ptr[idx] * rsqrt_rms_sum;idx+=BLOCK_SIZE;
        output_ptr[idx] = input_ptr[idx] * rsqrt_rms_sum;idx+=BLOCK_SIZE;
        output_ptr[idx] = input_ptr[idx] * rsqrt_rms_sum;idx+=BLOCK_SIZE;
    }

}


template <typename T, int BLOCK_SIZE = 256, int PADDING_LENGTH=4096> // == blk_size
__global__ void rms_unroll_vector4(const T *input, T *output)
{
    int seq_offset = blockIdx.x * PADDING_LENGTH;
    float *input_ptr = (float *)input + seq_offset;
    float *output_ptr = (float *)output + seq_offset;

    int tid = threadIdx.x;
    __shared__ float maxreduce_data[BLOCK_SIZE];

    // Find the maximum value in the block
    float max_tid = 0;
    int offset = tid*4;
    // Note that the LG Throttle is slightly(really slight) high in ncu
    // so we use vector memory access to release this throttle
    // but this will increase the risk of other instructions
    // and result in more LG Throttles
    {
        float4 vec_4 = *((float4*)(input_ptr+offset));
        max_tid +=vec_4.x * vec_4.x;
        max_tid +=vec_4.y * vec_4.y;
        max_tid +=vec_4.w * vec_4.w;
        max_tid +=vec_4.z * vec_4.z;
        offset+=4*BLOCK_SIZE;
    }
    {
        float4 vec_4 = *((float4*)(input_ptr+offset));
        max_tid +=vec_4.x * vec_4.x;
        max_tid +=vec_4.y * vec_4.y;
        max_tid +=vec_4.w * vec_4.w;
        max_tid +=vec_4.z * vec_4.z;
        offset+=4*BLOCK_SIZE;
    }
    {
        float4 vec_4 = *((float4*)(input_ptr+offset));
        max_tid +=vec_4.x * vec_4.x;
        max_tid +=vec_4.y * vec_4.y;
        max_tid +=vec_4.w * vec_4.w;
        max_tid +=vec_4.z * vec_4.z;
        offset+=4*BLOCK_SIZE;
    }
    {
        float4 vec_4 = *((float4*)(input_ptr+offset));
        max_tid +=vec_4.x * vec_4.x;
        max_tid +=vec_4.y * vec_4.y;
        max_tid +=vec_4.w * vec_4.w;
        max_tid +=vec_4.z * vec_4.z;
    }

    maxreduce_data[tid] = max_tid;
    __syncthreads();

    if (BLOCK_SIZE >= 256)
    {
        if (tid < 128)
        {
            maxreduce_data[tid] += maxreduce_data[tid + 128];
        }
        __syncthreads();
    }

    if (BLOCK_SIZE >= 128)
    {
        if (tid < 64)
        {
            maxreduce_data[tid] += maxreduce_data[tid + 64];
        }
        __syncthreads();
    }

    if (tid < 32)
    {
        warpReduceSum<BLOCK_SIZE>(maxreduce_data, tid);
    }
    __syncthreads();

    float pow_sum = maxreduce_data[0];
    float rsqrt_rms_sum = rsqrtf(pow_sum/PADDING_LENGTH);// avoid divide instructions and rsqrt has faster inplementation, also this decrease the arithmetic intensity.
    

    offset = tid*4;
    {
        float4* vec_4 = (float4*)(input_ptr+offset);
        float4* output_vec_4 = (float4*)(output_ptr+offset);
        output_vec_4->x=vec_4->x * rsqrt_rms_sum;
        output_vec_4->y=vec_4->y * rsqrt_rms_sum;
        output_vec_4->z=vec_4->z * rsqrt_rms_sum;
        output_vec_4->w=vec_4->w * rsqrt_rms_sum;
        offset+=4*BLOCK_SIZE;
    }
    {
        float4* vec_4 = (float4*)(input_ptr+offset);
        float4* output_vec_4 = (float4*)(output_ptr+offset);
        output_vec_4->x=vec_4->x * rsqrt_rms_sum;
        output_vec_4->y=vec_4->y * rsqrt_rms_sum;
        output_vec_4->z=vec_4->z * rsqrt_rms_sum;
        output_vec_4->w=vec_4->w * rsqrt_rms_sum;
        offset+=4*BLOCK_SIZE;
    }
    {
        float4* vec_4 = (float4*)(input_ptr+offset);
        float4* output_vec_4 = (float4*)(output_ptr+offset);
        output_vec_4->x=vec_4->x * rsqrt_rms_sum;
        output_vec_4->y=vec_4->y * rsqrt_rms_sum;
        output_vec_4->z=vec_4->z * rsqrt_rms_sum;
        output_vec_4->w=vec_4->w * rsqrt_rms_sum;
        offset+=4*BLOCK_SIZE;
    }
    {
        float4* vec_4 = (float4*)(input_ptr+offset);
        float4* output_vec_4 = (float4*)(output_ptr+offset);
        output_vec_4->x=vec_4->x * rsqrt_rms_sum;
        output_vec_4->y=vec_4->y * rsqrt_rms_sum;
        output_vec_4->z=vec_4->z * rsqrt_rms_sum;
        output_vec_4->w=vec_4->w * rsqrt_rms_sum;
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
    "ggml rms_norm_f32",
    [&](cudaStream_t s)
    {
        constexpr int block_width = 256;
        dim3 grid_size(seq_len);
        dim3 block_size(block_width);
        rms_norm_f32<block_width><<<grid_size, block_size, 0, s>>>(
            static_cast<const T *>(input.data_ptr()), baseline.data_ptr(), head_embed, 0.f);
    },
    [&](cudaStream_t s){
        return true;
    });

    state.run(
    "rms_norm_native",
    [&](cudaStream_t s)
    {
        constexpr int block_width = 256;
        dim3 grid_size(seq_len);
        dim3 block_size(block_width);
        rms_native<T, block_width><<<grid_size, block_size, 0, s>>>(
            static_cast<const T *>(input.data_ptr()), output.data_ptr(), head_embed);
    },
    VERIFY_FUNC);

    state.run(
    "rms_norm_unroll",
    [&](cudaStream_t s)
    {
        constexpr int block_width = 256;
        dim3 grid_size(seq_len);
        dim3 block_size(block_width);
        rms_unroll<T, block_width, head_embed><<<grid_size, block_size, 0, s>>>(
            static_cast<const T *>(input.data_ptr()), output.data_ptr());
    },
    VERIFY_FUNC);

    state.run(
    "rms_unroll_vector4",
    [&](cudaStream_t s)
    {
        constexpr int block_width = 256;
        dim3 grid_size(seq_len);
        dim3 block_size(block_width);
        rms_unroll_vector4<T, block_width, head_embed><<<grid_size, block_size, 0, s>>>(
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
