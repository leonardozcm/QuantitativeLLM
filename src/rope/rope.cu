#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include "qttbench/qtt_state.h"
#include "qttbench/qtt_tensor.cuh"

__global__ void verify_gpu(const float *c, const float *a, int seq_len, int *ret, const int HD=128)
{
    if (!(*ret))
        return;
    int hd_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int seq_idx = blockDim.y * blockIdx.y + threadIdx.y;
    int head_idx = blockDim.z * blockIdx.z + threadIdx.z;

    int idx = head_idx * seq_len * HD + HD * seq_idx + hd_idx;
    // printf("%f, %f\n", c[idx], a[idx]);
    // race but matters not
    if ((*ret) &&
        // This is not a good sanity check method, but in this experiment this is good enough.
        // refactor it with reduce sum mean diff
        fabs(a[idx]) > 0.001 && fabs((c[idx] - a[idx]) / a[idx]) > 0.02)
    {
        printf("%f %f\n", c[idx], a[idx]);
        (*ret) = 0;
    }
}

template <typename T>
__global__ void reset_gpu(T *m_t, const int seq_len, const int HD=128)
{
    int hd_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int seq_idx = blockDim.y * blockIdx.y + threadIdx.y;
    int head_idx = blockDim.z * blockIdx.z + threadIdx.z;

    int idx = head_idx * seq_len * HD + HD * seq_idx + hd_idx;
    m_t[idx] = 0.f;
}

#define VERIFY_FUNC                                                                                                        \
    [&](cudaStream_t s)                                                                                                    \
    {                                                                                                                      \
        constexpr int block_width = 16;                                                                                    \
        constexpr int block_height = 16;                                                                                   \
        const size_t head_dim = output.ne[0];                                                                                 \
        const size_t seq_len = output.ne[2];                                                                                  \
        const size_t num_heads = output.ne[1];                                                                                \
        const size_t bs = output.ne[3];                                                                                       \
        dim3 grid_size(head_dim / block_width, seq_len / block_height, bs * num_heads);                                    \
        dim3 block_size(block_width, block_height, 1);                                                                     \
        int *d_correct;                                                                                                    \
        int correct = 1;                                                                                                   \
        cudaMalloc(&d_correct, sizeof(int));                                                                               \
        cudaMemcpy(d_correct, &correct, sizeof(int), cudaMemcpyHostToDevice);                                              \
        verify_gpu<<<grid_size, block_size, 0, s>>>(output.data_ptr(), baseline.data_ptr(), seq_len, d_correct); \
        cudaMemcpy(&correct, d_correct, sizeof(int), cudaMemcpyDeviceToHost);                                              \
        reset_gpu<T><<<grid_size, block_size, 0, s>>>(output.data_ptr(), seq_len);                               \
        return correct;                                                                                                    \
    }

// Kernels
//
// Consider case that seq_id starts from 1 to seq_len
// q/k shape is [bs, seq_len, num_heads, HD]
//
// let x[seq_idx][idx] = q[:, seq_id, :, idx]
// let f_cos/sin(seq_idx, idx) = cos/sin(seq_idx/(10000 ^ (2* (idx%(HD/2)/HD))))
// then rope formula denotes to:
//
// if idx< HD/2, ROPE(x[seq_idx][idx]) = x[seq_idx][idx] * f_cos(seq_idx, idx) - x[idx+HD/2] * f_sin(seq_idx, idx)
// if idx>=HD/2, ROPE(x[seq_idx][idx]) = x[seq_idx][idx] * f_cos(seq_idx, idx) + x[idx-HD/2] * f_sin(seq_idx, idx)
__device__ __forceinline__ float f_sin(int seq_idx, int idx, int HD=128)
{
    float e = 2.f * ((float)(idx % (HD / 2)) / HD);
    // pow will invoke double precision ops
    // use powf will only invoke single precision
    return sinf(seq_idx / powf(10000, e));
}

__device__ float f_cos(int seq_idx, int idx, int HD=128)
{
    return cosf(seq_idx / (powf(10000, (2.f * ((float)(idx % (HD / 2)) / HD)))));
}

template <typename T>
__global__ void rope_native(const T *input, T *output, const size_t s1, const size_t s2)
{
    int hd_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int seq_idx = blockDim.y * blockIdx.y + threadIdx.y;
    int head_idx = blockDim.z * blockIdx.z + threadIdx.z;
    int HD = s1;

    // assert bs == 1
    int idx = head_idx * s1 + s2 * seq_idx + hd_idx;

    if (hd_idx < (HD >> 1))
    {
        output[idx] = input[idx] * f_cos(seq_idx, hd_idx) - input[idx + (HD >> 1)] * f_sin(seq_idx, hd_idx);
    }
    else
    {
        output[idx] = input[idx] * f_cos(seq_idx, hd_idx) + input[idx - (HD >> 1)] * f_sin(seq_idx, hd_idx);
    }
}

template <typename T>
__global__ void rope_native_x2(const T *input, T *output, const size_t s1, const size_t s2)
{
    int hd_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int seq_idx = blockDim.y * blockIdx.y + threadIdx.y;
    int head_idx = blockDim.z * blockIdx.z + threadIdx.z;

    // assert bs == 1
    int idx = head_idx * s1 + s2 * seq_idx + hd_idx;
    int HD = s1;

    float input_0 = input[idx];
    float input_1 = input[idx + (HD >> 1)];
    float cos = f_cos(seq_idx, hd_idx);
    float sin = f_sin(seq_idx, hd_idx);

    output[idx] = input_0 * cos - input_1 * sin;
    output[idx + (HD >> 1)] = input_1 * cos + input_0 * sin;
}

template <typename T>
__global__ void rope_calculate_once(const T *input, T *output, const size_t s1, const size_t s2)
{
    int hd_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int seq_idx = blockDim.y * blockIdx.y + threadIdx.y;
    int head_idx = blockDim.z * blockIdx.z + threadIdx.z;

    // assert bs == 1
    int idx = head_idx * s1 + s2 * seq_idx + hd_idx;
    int HD = s1;

    float input_0 = input[idx];
    float input_1 = input[idx + (HD >> 1)];
    // float e = 2.f * ((float)(idx % (HD / 2)) / HD);
    float pow_e = seq_idx / powf(10000, 2.f * ((float)(idx % (HD / 2)) / HD));
    float cos = cosf(pow_e);
    float sin = sinf(pow_e);

    output[idx] = input_0 * cos - input_1 * sin;
    output[idx + (HD >> 1)] = input_1 * cos + input_0 * sin;
}

// Copied from https://github.com/ggerganov/llama.cpp/blob/19d3c8293b1f61acbe2dab1d49a17950fd788a4a/ggml/src/ggml-cuda/rope.cu
//
// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
template <bool forward>
static __device__ void rope_yarn(
    const float theta_extrap, const float freq_scale, const int64_t i0, const float ext_factor,
    float mscale, float &cos_theta, float &sin_theta)
{
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;

    cos_theta = cosf(theta) * mscale;
    sin_theta = sinf(theta) * mscale;
    if (!forward)
    {
        sin_theta *= -1.0f;
    }
}

template <bool forward, bool has_ff, typename T>
static __global__ void rope_neox(
    const T *x, T *dst, const int ne0, const int ne1, const int s1, const int s2, const int n_dims,
    const int32_t *pos, const float freq_scale, const float ext_factor, const float attn_factor,
    const float theta_scale, const float *freq_factors)
{
    const int i0 = 2 * (blockDim.y * blockIdx.y + threadIdx.y);

    if (i0 >= ne0)
    {
        return;
    }

    const int row_dst = blockDim.x * blockIdx.x + threadIdx.x;

    if (i0 >= n_dims)
    {
        const int i = row_dst * ne0 + i0;

        dst[i + 0] = x[i + 0];
        dst[i + 1] = x[i + 1];

        return;
    }

    const int row_x = row_dst % ne1;
    const int channel_x = row_dst / ne1;

    const int idst = row_dst * ne0 + i0 / 2;
    const int ix = channel_x * s2 + row_x * s1 + i0 / 2;

    const float theta_base = pos[channel_x] * powf(theta_scale, i0 / 2.0f);

    const float freq_factor = has_ff ? freq_factors[i0 / 2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, i0, ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = x[ix + 0];
    const float x1 = x[ix + n_dims / 2];

    dst[idst + 0] = x0 * cos_theta - x1 * sin_theta;
    dst[idst + n_dims / 2] = x0 * sin_theta + x1 * cos_theta;
}

template <typename T>
void test_with_dtype(qttbench::State &state)
{
    // constexpr size_t N = 1024 * 16;
    // size_t size = N * N * sizeof(T);
    // T *a = (T *)malloc(size);

    constexpr int bs = 1, num_heads = 32, seq_len = 1024, head_dim = 128;

    std::vector<int> ne = {head_dim, num_heads, seq_len, bs};

    qttbench::Tensor<qttbench::float32_t> input(4, ne);
    input.initialize_random();

    qttbench::Tensor<qttbench::float32_t> baseline(4, ne);
    qttbench::Tensor<qttbench::float32_t> output(4, ne);

    int32_t *pos_cpu = (int32_t *)malloc(seq_len * sizeof(int32_t));
    for (int i = 0; i < seq_len; i++)
    {
        pos_cpu[i] = i;
    }
    qttbench::Tensor<qttbench::int32_t> pos(1, {seq_len}, pos_cpu, true);
    free(pos_cpu);

    state.run(
        "Llama.cpp rope neox",
        [&](cudaStream_t s)
        {
            constexpr int CUDA_ROPE_BLOCK_SIZE = 256;
            const size_t ne0 = input.ne[0];
            const size_t ne1 = input.ne[1];
            const size_t s1 = input.nb[1]/ input.nb[0];
            const size_t s2 = input.nb[2]/ input.nb[0];
            const size_t n_dims = ne0;

            const float freq_scale = 1.0f;
            const float ext_factor = 1.0f; // unused
            const float freq_base = 10000;
            const float attn_factor = 1.0f;
            const float theta_scale = powf(freq_base, -2.0f / n_dims);
            float *freq_factors = nullptr;

            const int nr = input.nb[3] / input.nb[1];

            const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
            const int n_blocks_x = (ne0 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
            const dim3 block_nums(nr, n_blocks_x, 1);
            rope_neox<true, false, T><<<block_nums, block_dims, 0, s>>>(
                input.data_ptr(), baseline.data_ptr(), ne0, ne1, s1, s2, n_dims,
                pos.data_ptr(), freq_scale, ext_factor,
                attn_factor, theta_scale, freq_factors);
        },
        [&](cudaStream_t s)
        {
            // baseline
            return 1;
        });

    state.run(
        "rope_native blocksize 16 * 16",
        [&](cudaStream_t s)
        {
            constexpr int block_width = 16;
            constexpr int block_height = 16;
            const size_t head_dim = input.ne[0];
            const size_t seq_len = input.ne[2];
            const size_t num_heads = input.ne[1];
            const size_t bs = input.ne[3];

            const size_t s1 = input.nb[1]/ input.nb[0];
            const size_t s2 = input.nb[2]/ input.nb[0];
            dim3 grid_size(head_dim / block_width, seq_len / block_height, bs * num_heads);
            dim3 block_size(block_width, block_height, 1);
            rope_native<T><<<grid_size, block_size, 0, s>>>(
                static_cast<const T*>(input.data_ptr()), output.data_ptr(), s1, s2);
        },
        VERIFY_FUNC);

    state.run(
        "rope merge x2",
        [&](cudaStream_t s)
        {
            constexpr int block_width = 8;
            constexpr int block_height = 16;
            const size_t head_dim = input.ne[0];
            const size_t seq_len = input.ne[2];
            const size_t num_heads = input.ne[1];
            const size_t bs = input.ne[3];

            const size_t s1 = input.nb[1]/ input.nb[0];
            const size_t s2 = input.nb[2]/ input.nb[0];
            dim3 grid_size(head_dim / block_width / 2, seq_len / block_height, bs * num_heads);
            dim3 block_size(block_width, block_height, 1);
            rope_native_x2<T><<<grid_size, block_size, 0, s>>>(
                input.data_ptr(), output.data_ptr(), s1, s2);
        },
        VERIFY_FUNC);

    state.run(
        "rope calculate_once",
        [&](cudaStream_t s)
        {
            constexpr int block_width = 8;
            constexpr int block_height = 16;
            const size_t head_dim = input.ne[0];
            const size_t seq_len = input.ne[2];
            const size_t num_heads = input.ne[1];
            const size_t bs = input.ne[3];

            const size_t s1 = input.nb[1]/ input.nb[0];
            const size_t s2 = input.nb[2]/ input.nb[0];
            dim3 grid_size(head_dim / block_width / 2, seq_len / block_height, bs * num_heads);
            dim3 block_size(block_width, block_height, 1);
            rope_calculate_once<T><<<grid_size, block_size, 0, s>>>(
                input.data_ptr(), output.data_ptr(), s1, s2);
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
