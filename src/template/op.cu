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


template <typename T>
__global__ void op(const T *input, T *output, const size_t s1, const size_t s2)
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
void test_with_dtype(qttbench::State &state)
{
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
        "op_native",
        [&](cudaStream_t s)
        {
            dim3 grid_size(gx, gy, gw);
            dim3 block_size(bx, by, bw);
            op<T><<<grid_size, block_size, 0, s>>>(
                static_cast<const T*>(input.data_ptr()), output.data_ptr(), s1, s2);
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
