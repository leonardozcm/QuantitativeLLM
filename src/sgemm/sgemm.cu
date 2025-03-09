#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <cublas_v2.h>
#include "qttbench/qtt_state.h"
#include "qttbench/qtt_tensor.cuh"

#define WARP_SIZE 32

__global__ void verify_gpu(const float *C, const float *baseline, int *ret)
{
    if (!(*ret))
        return;

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // printf("%f, %f\n", c[idx], a[idx]);
    // race but matters not
    // printf("idx:[%d, %d] %f %f\n", idx/256, idx%256, C[idx], baseline[idx]);
    if ((*ret) &&
        // This is not a good sanity check method, but in this experiment this is good enough.
        // refactor it with reduce sum mean diff
        (fabs(baseline[idx]) > 0.001 || fabs(C[idx]) > 0.01) &&
        fabs((C[idx] - baseline[idx]) / fmax(baseline[idx], C[idx])) > 0.05)
    {
        printf("idx:[%d, %d] %f %f\n", idx / 256, idx % 256, C[idx], baseline[idx]);
        (*ret) = 0;
    }
}

template <typename T>
__global__ void reset_gpu(T *m_t, const int HD = 128)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    m_t[idx] = 0.f;
}

#define VERIFY_FUNC                                                                                \
    [&](cudaStream_t s)                                                                            \
    {                                                                                              \
        constexpr int block_width = 64;                                                            \
        const size_t n_elements = baseline.n_elements();                                           \
        dim3 grid_size(n_elements / block_width, 1, 1);                                            \
        dim3 block_size(block_width, 1, 1);                                                        \
        int *d_correct;                                                                            \
        int correct = 1;                                                                           \
        cudaMalloc(&d_correct, sizeof(int));                                                       \
        cudaMemcpy(d_correct, &correct, sizeof(int), cudaMemcpyHostToDevice);                      \
        verify_gpu<<<grid_size, block_size, 0, s>>>(C.data_ptr(), baseline.data_ptr(), d_correct); \
        cudaMemcpy(&correct, d_correct, sizeof(int), cudaMemcpyDeviceToHost);                      \
        reset_gpu<T><<<grid_size, block_size, 0, s>>>(C.data_ptr());                               \
        return correct;                                                                            \
    }

template <int BLOCK_SIZE = 256>
__global__ void sgemm_native(const float *A, const float *B, float *C, const size_t M, const size_t K, const size_t N)
{
    int tid = threadIdx.x;
    int gid_x = blockIdx.x * BLOCK_SIZE;
    int gid_y = blockIdx.y;

    const float *A_ptr = A + gid_y * K;
    const float *B_ptr = B + (gid_x + tid) * K;
    float *C_ptr = C + gid_y * N + gid_x + tid;

    float res = 0;
    for (int i = 0; i < K; i++)
    {
        res += A_ptr[i] * B_ptr[i];
    }
    *C_ptr = res;
}

/** 
 * with microbenchmark provided by https://github.com/Yinghan-Li/YHs_Sample/tree/master/cuda/microbenchmark
 * For RTX2070s:
 * 
 * DRAM latency 471 cycles
 * L2 cache latency 214 cycles
 * L1 cache latency 32 cycles
 * smem latency 22 cycles 
 * shared memory bandwidth per SM (measured): 111.734879 byte/cycle
 * shared memory bandwidth per SM (theoretical): 128 byte/cycle
 * 
 * DRAM bandwidth
```
 4MB (r+w)
read 339.508581GB/s
write 349.381258GB/s
copy 347.432216GB/s
---------------------------
8MB (r+w)
read 292.987497GB/s
write 336.411597GB/s
copy 326.229843GB/s
---------------------------
16MB (r+w)
read 302.639292GB/s
write 342.583800GB/s
copy 323.416465GB/s
---------------------------
```
 * L2 cache bandwidth 1811.228600GB/s
 * smem cache bandwidth 9139.200195 GB/s
 * */
template <int BLOCK_SIZE = 256>
__global__ void sgemm_tiling_base(const float *A, const float *B, float *C, const size_t M, const size_t K, const size_t N)
{
    int tid = threadIdx.x;
    int gid_x = blockIdx.x * BLOCK_SIZE;
    int gid_y = blockIdx.y;

    const float *A_ptr = A + gid_y * K;
    const float *B_ptr = B + (gid_x + tid) * K;
    float *C_ptr = C + gid_y * N + gid_x + tid;

    float res = 0;
    for (int i = 0; i < K; i++)
    {
        res += A_ptr[i] * B_ptr[i];
    }
    *C_ptr = res;
}

static __device__ __forceinline__ float warp_reduce_sum(float x)
{
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        x += __shfl_xor_sync(0xffffffff, x, offset, 32);
    }
    return x;
}


template <typename T, int SL = 1024, int HA = 4096, int HB = 4096>
void test_with_dtype(qttbench::State &state)
{
    constexpr int bs = 1, seq_len = SL, hidden_status_A = HA, hidden_status_B = HB;

    qttbench::Tensor<qttbench::float32_t> A(3, {hidden_status_A, seq_len, bs});
    qttbench::Tensor<qttbench::float32_t> B(3, {hidden_status_A, hidden_status_B, bs});
    // A.initialize_random();
    // B.initialize_random();

    qttbench::Tensor<qttbench::float32_t> baseline(3, {hidden_status_B, seq_len, bs});
    qttbench::Tensor<qttbench::float32_t> C(3, {hidden_status_B, seq_len, bs});

    state.run(
        "sgemm_cublas",
        [&](cudaStream_t s)
        {
            const int K = hidden_status_A;
            const int N = hidden_status_B;
            const int M = seq_len;
            // refer to https://github.com/NVIDIA/CUDALibrarySamples/tree/4125fcded1ff466efe73e82e1ef98ef9572fbc84/cuBLAS/Level-2/gemv
            // Construct cublas handle
            cublasHandle_t cublasH = NULL;
            cublasCreate(&cublasH);
            const float alpha = 1.0;
            const float beta = 0.0;

            cublasSgemm(
                cublasH,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                N, M, K,
                &alpha,
                B.data_ptr(), K,
                A.data_ptr(), K,
                &beta,
                baseline.data_ptr(), N
            );

        },
        [&](cudaStream_t s)
        {
            return true;
        });

    state.run(
        "sgemm_native",
        [&](cudaStream_t s)
        {
            const size_t K = hidden_status_A;
            const size_t N = hidden_status_B;
            const size_t M = seq_len;
            const size_t ne2 = bs;
            constexpr int BLOCK_SIZE = 256;

            dim3 grid_size((N + BLOCK_SIZE - 1) / BLOCK_SIZE, M, ne2);
            dim3 block_size(BLOCK_SIZE, 1, 1);
            sgemm_native<BLOCK_SIZE><<<grid_size, block_size, 0, s>>>(
                A.data_ptr(), B.data_ptr(), C.data_ptr(), M, K, N);
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

    // test_with_dtype<qttbench::float32_t, 64, 64>(state);
    test_with_dtype<qttbench::float32_t, 2, 4096, 4096>(state);
    // test_with_dtype<qttbench::float32_t, 2, 4096, 14336>(state);
    // test_with_dtype<qttbench::float32_t, 4096, 14336>(state);
    return 0;
}
