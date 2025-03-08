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
    // printf("idx:[%d, %d] %f %f\n", idx/256, idx%256, C[idx], baseline[idx]);
    if ((*ret) &&
    // This is not a good sanity check method, but in this experiment this is good enough.
    // refactor it with reduce sum mean diff
    (fabs(baseline[idx]) > 0.001 || fabs(C[idx])>0.01) &&
    fabs((C[idx] - baseline[idx]) / fmax(baseline[idx], C[idx])) > 0.05)
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
        constexpr int block_width = 64;                                                                \
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

template<int BLOCK_SIZE=256>
__global__ void sgemv_native(const float *A, const float *B, float *C, const size_t K)
{
    int tid = threadIdx.x;
    int gid = blockIdx.x * BLOCK_SIZE;
    
    const float* B_ptr = B + (gid + tid)*K;
    float* C_ptr = C + gid;

    float res=0;
    for(int i=0; i<K; i++){
        res+=A[i]*B_ptr[i];
    }
    C_ptr[tid] = res;
}


static __device__ __forceinline__ float warp_reduce_sum(float x)
{
#pragma unroll
    for(int offset = 16; offset>0; offset>>=1)
    {
        x+= __shfl_xor_sync(0xffffffff,x,offset, 32);
    }
    return x;
}

#define WARP_SIZE 32
// assert K mod blockDim.x == 0
template<int N_BLOCK, int K>
__global__ void sgemv_tiling_base(const float *A, const float *B, float *C)
{
    __shared__ float A_slm[K];
    const int tx = blockDim.x;
    const int tid = threadIdx.x;
    
    // load A to slm
    #pragma unroll
    for(int i=tid; i<K; i+=tx){
        A_slm[i]=A[i];
    }
    __syncthreads();
    
    const int warp_idx = tid / WARP_SIZE;
    const int lane_idx = tid % WARP_SIZE;
    const int warps_num = tx / WARP_SIZE;

    const float* B_blk_base_t = B + (blockIdx.x*N_BLOCK)*K;
    
    #pragma unroll
    for(int i=warp_idx; i<N_BLOCK; i+=warps_num){
        const float* B_warp_t = B_blk_base_t + i*K;
        float sum = 0.f;
        for(int j=lane_idx; j<K; j+=WARP_SIZE){
            sum+=B_warp_t[j]*A_slm[j];
        }
        sum=warp_reduce_sum(sum);
        if(lane_idx==0)C[blockIdx.x*N_BLOCK+i]=sum;
    }
}

template<int N_BLOCK, int K>
__global__ void sgemv_tiling_no_slm(const float *A, const float *B, float *C)
{
    // __shared__ float A_slm[K];
    const int tx = blockDim.x;
    const int tid = threadIdx.x;
    
    const int warp_idx = tid / WARP_SIZE;
    const int lane_idx = tid % WARP_SIZE;
    const int warps_num = tx / WARP_SIZE;

    const float* B_blk_base_t = B + (blockIdx.x*N_BLOCK)*K;
    
    #pragma unroll
    for(int i=warp_idx; i<N_BLOCK; i+=warps_num){
        const float* B_warp_t = B_blk_base_t + i*K;
        float sum = 0.f;
        for(int j=lane_idx; j<K; j+=WARP_SIZE){
            sum+=B_warp_t[j]*A[j];
        }
        sum=warp_reduce_sum(sum);
        if(lane_idx==0)C[blockIdx.x*N_BLOCK+i]=sum;
    }
}

// on case 4096 * 4096
// GPU SMs can not be saturated by kernels above
// if we only have 32 blocks to achieve this work
// so in this kernel we have blocks with less work to do and more blocks cooperate at the same time
template<int N_BLOCK, int K>
__global__ void sgemv_saturate_sm(const float *A, const float *B, float *C)
{
    __shared__ float A_slm[N_BLOCK * WARP_SIZE];
    const int warp_idx = threadIdx.y;
    const int lane_idx = threadIdx.x;
    const int blk_idx = blockIdx.x;
    const int tid = warp_idx*WARP_SIZE+lane_idx;

    const float* A_blk_base_t = A;
    const float* B_blk_base_t = B + (blk_idx*N_BLOCK+warp_idx)*K;

    float sum = 0.f;
    #pragma unroll
    for(int i=lane_idx; i<K; i+=WARP_SIZE*N_BLOCK){
        // load A shard to slm
        A_slm[tid] = A_blk_base_t[tid];
        __syncthreads();
        
        #pragma unroll
        for(int j=lane_idx; j<N_BLOCK*WARP_SIZE; j+=WARP_SIZE){
            sum+=A_slm[j]*B_blk_base_t[j];
        }
        A_blk_base_t+=WARP_SIZE*N_BLOCK;
        B_blk_base_t += WARP_SIZE*N_BLOCK;
    }

    sum=warp_reduce_sum(sum);
    if(lane_idx==0)C[blk_idx*N_BLOCK+warp_idx]=sum;

}

template <typename T, int HA=4096, int HB=4096>
void test_with_dtype(qttbench::State &state)
{
    constexpr int bs = 1, seq_len = 1, hidden_status_A=HA, hidden_status_B=HB;

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

            // cublasSgemm(
            //     cublasH,
            //     CUBLAS_OP_T,
            //     CUBLAS_OP_N,
            //     1, ne0_B,  ne0_A,
            //     &alpha,
            //     A.data_ptr(), ne0_A,
            //     B.data_ptr(), ne0_A,
            //     &beta,
            //     baseline.data_ptr(), ne0_B
            // );
            
            // B^T * vec(A) = C^T
            // for C is a vector so transposed doesn't change the memory layout.
            //
            cublasSgemv(
                cublasH,
                CUBLAS_OP_T,
                ne0_A, ne0_B,
                &alpha,
                B.data_ptr(), ne0_A,
                A.data_ptr(), 1,
                &beta,
                baseline.data_ptr(), 1
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

        state.run(
        "sgemv_tiling_base 64 128",
        [&](cudaStream_t s)
        {
            constexpr size_t ne0_A = hidden_status_A;
            const size_t ne0_B = hidden_status_B;
            const size_t ne1 = seq_len;
            const size_t ne2 = bs;
            constexpr int BLOCK_SIZE = 128;
            constexpr int tx = 64;

            dim3 grid_size((ne0_B+BLOCK_SIZE-1)/BLOCK_SIZE, ne1, ne2);
            dim3 block_size(tx, 1, 1);
            sgemv_tiling_base<BLOCK_SIZE, ne0_A><<<grid_size, block_size, 0, s>>>(
                A.data_ptr(), B.data_ptr(), C.data_ptr());
        },
        VERIFY_FUNC);

        state.run(
        "sgemv_tiling_base 64 256",
        [&](cudaStream_t s)
        {
            constexpr size_t ne0_A = hidden_status_A;
            const size_t ne0_B = hidden_status_B;
            const size_t ne1 = seq_len;
            const size_t ne2 = bs;
            constexpr int BLOCK_SIZE = 256;
            constexpr int tx = 64;

            dim3 grid_size((ne0_B+BLOCK_SIZE-1)/BLOCK_SIZE, ne1, ne2);
            dim3 block_size(tx, 1, 1);
            sgemv_tiling_base<BLOCK_SIZE, ne0_A><<<grid_size, block_size, 0, s>>>(
                A.data_ptr(), B.data_ptr(), C.data_ptr());
        },
        VERIFY_FUNC);

        state.run(
        "sgemv_tiling_base 128 256",
        [&](cudaStream_t s)
        {
            constexpr size_t ne0_A = hidden_status_A;
            const size_t ne0_B = hidden_status_B;
            const size_t ne1 = seq_len;
            const size_t ne2 = bs;
            constexpr int BLOCK_SIZE = 256;
            constexpr int tx = 128;

            dim3 grid_size((ne0_B+BLOCK_SIZE-1)/BLOCK_SIZE, ne1, ne2);
            dim3 block_size(tx, 1, 1);
            sgemv_tiling_base<BLOCK_SIZE, ne0_A><<<grid_size, block_size, 0, s>>>(
                A.data_ptr(), B.data_ptr(), C.data_ptr());
        },
        VERIFY_FUNC);

        state.run(
        "sgemv_saturate_sm BLOCK SIZE 4",
        [&](cudaStream_t s)
        {
            constexpr size_t ne0_A = hidden_status_A;
            const size_t ne0_B = hidden_status_B;
            const size_t ne1 = seq_len;
            const size_t ne2 = bs;
            constexpr int BLOCK_SIZE = 4;

            dim3 grid_size((ne0_B+BLOCK_SIZE-1)/BLOCK_SIZE, ne1, ne2);
            dim3 block_size(WARP_SIZE, BLOCK_SIZE, 1);
            sgemv_saturate_sm<BLOCK_SIZE, ne0_A><<<grid_size, block_size, 0, s>>>(
                A.data_ptr(), B.data_ptr(), C.data_ptr());
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
    test_with_dtype<qttbench::float32_t>(state);
    // test_with_dtype<qttbench::float32_t, 14336>(state);
    // test_with_dtype<qttbench::float32_t, 4096, 14336>(state);
    return 0;
}
