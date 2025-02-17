#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
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
    if ((*ret) && c[x_id * N + y_id] != 2*a[x_id * N + y_id])
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
        verify_gpu<<<block_size, thread_size, 0, s>>>(d_c, d_a, N, N, d_correct); \
        cudaMemcpy(&correct, d_correct, sizeof(int), cudaMemcpyDeviceToHost);                      \
        reset_gpu<<<block_size, thread_size, 0, s>>>(d_c, N, N);                       \
        return correct;                                                                            \
    }


template<typename T>
__device__ __forceinline__ T op(T a, T b){
    return a + b;
}


// Kernels
// blocksize 1024(32 warps, max warps 48 per SM, so one SM can only put 1 block)
// utilize rate 32/48=66.7%
template<typename T>
__global__ void elementwiseop_native(const T* a, const T* b, T* c, int N){
    int col_idx=blockIdx.x*blockDim.x+threadIdx.x;
    int row_idx=blockIdx.y*blockDim.y+threadIdx.y;

    int idx = row_idx * N + col_idx;
    c[idx] = op(a[idx], b[idx]);
}

template<typename T>
__global__ void elementwiseop_float4(const T* a, const T* b, T* c, int N){
    int idx=(blockIdx.x*blockDim.x+threadIdx.x) << 2;
    const float4* a_4 = reinterpret_cast<const float4*>(a+idx);
    const float4* b_4 = reinterpret_cast<const float4*>(b+idx);
    float4* c_4 = reinterpret_cast<float4*>(c+idx);
    float4 reg_c;
    reg_c.x = op(a_4->x, b_4->x);
    reg_c.y = op(a_4->y, b_4->y);
    reg_c.z = op(a_4->z, b_4->z);
    reg_c.w = op(a_4->w, b_4->w);
    *(c_4) = reg_c;
}

template<typename T>
__global__ void elementwiseop_float_32x8(const T* a, const T* b, T* c, int N){
    int col_idx=blockIdx.x*blockDim.x+threadIdx.x;
    int row_idx=blockIdx.y*blockDim.y+threadIdx.y;

    int idx = row_idx * N * 8 + col_idx;
    #pragma unroll
    for(int i=0; i<8; i++){
        c[idx+i*N]=op(a[idx+i*N], b[idx+i*N]);
    }
}


template<typename T>
void test_with_dtype(qttbench::State& state){
    constexpr size_t N = 1024 * 16;
    size_t size = N * N * sizeof(T);
    T *a = (T *)malloc(size);
    T *b = (T *)malloc(size);
    T *c = (T *)malloc(size);

    // Init matrix
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            a[i * N + j] = i * N + j + 1;
            b[i * N + j] = i * N + j + 1;
        }
    }

    T *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    free(a);free(b);

    state.run(
        "elementwiseop_native blocksize 32*32",
        [&](cudaStream_t s)
        {
            constexpr int block_width=32;
            dim3 grid_size(N / block_width, N / block_width);
            dim3 block_size(block_width, block_width);
            elementwiseop_native<T><<<grid_size, block_size>>>(
                d_a, d_b, d_c, N
            );

        },
        VERIFY_FUNC
    );


    state.run(
        "elementwiseop_float4",
        [&](cudaStream_t s)
        {
            constexpr int block_width=256;
            constexpr int register_width = 4;
            dim3 grid_size(N * N / block_width / register_width);
            dim3 block_size(block_width);
            elementwiseop_float4<T><<<grid_size, block_size>>>(
                d_a, d_b, d_c, N
            );

        },
        VERIFY_FUNC
    );

    state.run(
        "elementwiseop_float 32x8",
        [&](cudaStream_t s)
        {
            constexpr int block_width=32;
            constexpr int block_height=8;
            constexpr int loop_size = 8;
            dim3 grid_size(N / block_width, N / block_height / loop_size);
            dim3 block_size(block_width, block_height);
            elementwiseop_float_32x8<T><<<grid_size, block_size>>>(
                d_a, d_b, d_c, N
            );

        },
        VERIFY_FUNC
    );

    cudaFree(d_a);cudaFree(d_b);cudaFree(d_c);
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