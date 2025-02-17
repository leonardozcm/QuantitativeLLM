#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include "qttbench/qtt_state.h"


template <typename T>
__global__ void verify_gpu(const T *c, const T *a, int M, int N, int *ret)
{
    if (!(*ret))
        return;
    const int b_x = blockIdx.x;
    const int b_y = blockIdx.y;
    const int t_x = threadIdx.x;
    const int t_y = threadIdx.y;
    const int x_id = b_x * blockDim.x + t_x;
    const int y_id = b_y * blockDim.y + t_y;

    // race but matters not
    if ((*ret) && c[x_id * N + y_id] != (a[x_id * N + y_id]>0?a[x_id * N + y_id]:0))
    {
        (*ret) = 0;
    }
}


template <typename T>
__global__ void reset_gpu(T *m_t, int M, int N)
{
    int b_x = blockIdx.x;
    int b_y = blockIdx.y;
    int t_x = threadIdx.x;
    int t_y = threadIdx.y;

    int x_id = b_x * blockDim.x + t_x;
    int y_id = b_y * blockDim.y + t_y;
    m_t[y_id * N + x_id] = 0.f;
}


#define VERIFY_FUNC                                                                                \
    [&](cudaStream_t s)                                                                            \
    {                                                                                              \
        dim3 block_size(N / 16, N / 16);                                                           \
        dim3 thread_size(16, 16);                                                                  \
        int *d_correct;                                                                            \
        int correct = 1;                                                                           \
        cudaMalloc(&d_correct, sizeof(int));                                                       \
        cudaMemcpy(d_correct, &correct, sizeof(int), cudaMemcpyHostToDevice);                      \
        verify_gpu<<<block_size, thread_size, 0, s>>>(d_b, d_a, N, N, d_correct); \
        cudaMemcpy(&correct, d_correct, sizeof(int), cudaMemcpyDeviceToHost);                      \
        reset_gpu<<<block_size, thread_size, 0, s>>>(d_b, N, N);                       \
        return correct;                                                                            \
    }


// Kernels
// blocksize 1024(32 warps, max warps 48 per SM, so one SM can only put 1 block)
// utilize rate 32/48=66.7%
template<typename T>
__global__ void relu_native(const T* a, T* b, int N){
    int col_idx=blockIdx.x*blockDim.x+threadIdx.x;
    int row_idx=blockIdx.y*blockDim.y+threadIdx.y;

    int idx = row_idx * N + col_idx;
    b[idx] = fmaxf(a[idx], 0);
}


template<typename T>
void test_with_dtype(qttbench::State& state){
    constexpr size_t N = 1024 * 16;
    size_t size = N * N * sizeof(T);
    T *a = (T *)malloc(size);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(-100, 100);

    // Init matrix
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            a[i * N + j] = dis(gen);
        }
    }

    T *d_a, *d_b;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

    free(a);

    state.run(
        "relu_native_native blocksize 32*8",
        [&](cudaStream_t s)
        {
            constexpr int block_width=32;
            constexpr int block_height=8;
            dim3 grid_size(N / block_width, N / block_height);
            dim3 block_size(block_width, block_height);
            relu_native<T><<<grid_size, block_size>>>(
                d_a, d_b, N
            );

        },
        VERIFY_FUNC
    );



    cudaFree(d_a);cudaFree(d_b);
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
            perf = std::stoi(perf_t)==0;
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
