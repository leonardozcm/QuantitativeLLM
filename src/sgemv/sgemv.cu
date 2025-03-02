#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <cublas_v2.h>
#include "qttbench/qtt_state.h"
#include "qttbench/qtt_tensor.cuh"

__global__ void verify_gpu(const float *C, const float *baseline, int *ret)
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
    fabs((C[idx] - baseline[idx]) / fmax(baseline[idx], C[idx])) > 0.02)
    {
        printf("idx:[%d, %d] %f %f\n", idx/256, idx%256, C[idx], baseline[idx]);
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
        verify_gpu<<<grid_size, block_size, 0, s>>>(C.data_ptr(), baseline.data_ptr(), d_correct); \
        cudaMemcpy(&correct, d_correct, sizeof(int), cudaMemcpyDeviceToHost);                           \
        reset_gpu<T><<<grid_size, block_size, 0, s>>>(C.data_ptr());                               \
        return correct;                                                                                 \
    }

template<int BLOCK_SIZE>
__global__ void sgemv_native(const float *A, const float *B, float *C, const size_t K)
{
    int tid = threadIdx.x;
    int gid = blockIdx.x + blockIdx.y*blockDim.x + blockIdx.z*blockDim.x*gridDim.y;
    
    const float* B_ptr = B + (gid*BLOCK_SIZE + tid)*K;
    float* C_ptr = C + gid;

    float res=0;
    for(int i=0; i<K; i++){
        res+=A[i]*B_ptr[i];
    }
    C_ptr[tid] = res;
}

template <typename T>
void test_with_dtype(qttbench::State &state)
{
    constexpr int bs = 1, seq_len = 1, hidden_status_A=256, hidden_status_B=256;

    qttbench::Tensor<qttbench::float32_t> A(3, {hidden_status_A, seq_len, bs});
    qttbench::Tensor<qttbench::float32_t> B(3, {hidden_status_A, hidden_status_B, bs});
    A.initialize_random();
    B.initialize_random();

    qttbench::Tensor<qttbench::float32_t> baseline(3, {hidden_status_B, seq_len, bs});
    qttbench::Tensor<qttbench::float32_t> C(3, {hidden_status_B, seq_len, bs});


    state.run(
        "sgemm_cublas",
        [&](cudaStream_t s)
        {
            const int ne0_A = hidden_status_A;
            const int ne0_B = hidden_status_B;
            // refer to https://github.com/NVIDIA/CUDALibrarySamples/tree/4125fcded1ff466efe73e82e1ef98ef9572fbc84/cuBLAS/Level-2/gemv
            // Construct cublas handle
            cublasHandle_t cublasH = NULL;
            cublasCreate(&cublasH);
            const float alpha = 1.0;
            const float beta = 0.0;

            // cublasSgemv(cublasH, transa, 1, ne0_B, &alpha, A.data_ptr(), lda, B.data_ptr(), ne0_A, &beta, baseline.data_ptr(), 1);
            cublasSgemm(
                cublasH,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                1, ne0_B,  ne0_A,
                &alpha,
                A.data_ptr(), ne0_A,
                B.data_ptr(), ne0_A,
                &beta,
                baseline.data_ptr(), ne0_B
            );
        },
        [&](cudaStream_t s)
        {
            return true;
        });

    state.run(
        "sgemv_native",
        [&](cudaStream_t s)
        {
            const size_t ne0_A = hidden_status_A;
            const size_t ne0_B = hidden_status_B;
            const size_t ne1 = seq_len;
            const size_t ne2 = bs;
            constexpr int BLOCK_SIZE = 256;

            dim3 grid_size((ne0_B+BLOCK_SIZE-1)/BLOCK_SIZE, ne1, ne2);
            dim3 block_size(BLOCK_SIZE, 1, 1);
            sgemv_native<BLOCK_SIZE><<<grid_size, block_size, 0, s>>>(
                A.data_ptr(), B.data_ptr(), C.data_ptr(), ne0_A);
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
