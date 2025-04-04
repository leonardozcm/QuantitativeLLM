#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <cublas_v2.h>
#include "qttbench/qtt_state.h"
#include "qttbench/qtt_tensor.cuh"

#define WARP_SIZE 32
#define DATA_THRESHOLD 10*0.1
#define ERR 10*0.005

__global__ void verify_gpu(const float *C, const float *baseline, int *ret)
{
    if (!(*ret))
        return;

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // printf("%f, %f\n", c[idx], a[idx]);
    // race but matters not
    bool flag =         (fabs(baseline[idx]) > DATA_THRESHOLD || fabs(C[idx]) > DATA_THRESHOLD) &&
    (fabs(C[idx] - baseline[idx]) > ERR);
    // printf("idx:[%d, %d] %f %f %s\n", idx / 128, idx % 128, C[idx], baseline[idx], flag?"false":"true");
    if ((*ret) &&
    // This is not a good sanity check method, but in this experiment this is good enough.
    // refactor it with reduce sum mean diff
    flag
    )
    {
    printf("idx:[%d, %d] %f %f %s\n", idx / 128, idx % 128, C[idx], baseline[idx], flag?"false":"true");
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

 // Gird_size<<<(N+N_BLOCK)/(N_BLOCK), (M+M_BLOCK)/M_BLOCK, 1>>>
 // block_size<<<16, 16, 1>>> tx, ty

 #define SIZE_OF_FLOAT4 32
 #define FLOAT4_NUMS 4
 #define FECTH_CONST_FLOAT4(x) *(reinterpret_cast<const float4*>(x))
 #define FECTH_FLOAT4(x) *(reinterpret_cast<float4*>(x))
template <int M_BLOCK=128, int N_BLOCK=128, int K_BLOCK=8, int M_THREAD=8, int N_THREAD=8>
__global__ void sgemm_tiling_base(const float *A, const float *B, float *C, const size_t M, const size_t K, const size_t N)
{
    const int tx=threadIdx.x;
    const int ty=threadIdx.y;
    const int blk_x=blockIdx.x;
    const int blk_y=blockIdx.y;

    int warp_idx = (tx+ty*blockDim.x)/WARP_SIZE; // 0~8
    // 0~2
    int warp_lane = (tx+ty*blockDim.x)%WARP_SIZE; // 0~32
    // 0~32
    const int warps_per_blk = blockDim.x * blockDim.y / WARP_SIZE; // 8
    // 2

    // for warps_blk_y ~ warps_blk_x covers most slm access latency
    const int warps_blk_y = (int)sqrtf(warps_per_blk); // 2
    // 1: major direction
    const int warps_blk_x = warps_per_blk / warps_blk_y; // 4
    // 2

    // For saving registers
    // assert(M_BLOCK==N_BLOCK) and (M_THREAD==N_THERAD)
    // so for loading A_blk and B_blk they share the same pattern
    constexpr int warp_k_ld_in_float4_per_row = K_BLOCK / FLOAT4_NUMS; // 8/4=2
    // 2
    const int thread_ld_offset_in_blk_y = M_BLOCK / warps_per_blk * warp_idx + warp_lane / warp_k_ld_in_float4_per_row; // 0~128
    // 128*(0,1)+0~16
    const int thread_ld_offset_in_blk_x = warp_lane % warp_k_ld_in_float4_per_row;  // 0 or 1
    // 0, 1
    constexpr int warp_ld_cover_y_in_blk = WARP_SIZE / warp_k_ld_in_float4_per_row; // 16
    // 16
    const int thread_ld_times = M_BLOCK / warps_per_blk / warp_ld_cover_y_in_blk; // 1
    // 8

    constexpr int slm_padding_dim = K_BLOCK+4;
    __shared__ float A_blk_tile[M_BLOCK][slm_padding_dim]; // avoid bank conflict, for SM Occupancy is limited by thread
    __shared__ float B_blk_tile[N_BLOCK][slm_padding_dim]; // avoid bank conflict, for SM Occupancy is limited by thread

    const float* A_blk_base_ptr = A+blk_y*M_BLOCK*K;
    const float* B_blk_base_ptr = B+blk_x*N_BLOCK*K;

    float A_thread_reg [M_THREAD] ={0.f};
    float B_thread_reg [M_THREAD] = {0.f};
    float C_thread_reg [M_THREAD][N_THREAD] ={0.f};
    const int warp_C_offset_in_blk_row = warp_idx / warps_blk_y; // 0~4
    // 0~2
    const int warp_C_offset_in_blk_column = warp_idx % warps_blk_y; // 0~2
    // 0
    const int warps_blk_A_size = M_BLOCK/warps_blk_x; // 32
    const int warps_blk_B_size = N_BLOCK/warps_blk_y; // 64
    const int A_slm_warp_offset = ty*M_THREAD; // 0, 8, 16, 24, 32, 40, 48, ... 120(8*15)
    const int B_slm_warp_offset = tx*N_THREAD; // 0, 8, 16, 24, 32, 40, 48, ... 120(8*15)

    // const int A_reg_ld_x_idx = tx%(warps_blk_A_size/M_THREAD)*M_THREAD;// 0, 8, 16, 24
    // const int B_reg_ld_y_idx = tx%(warps_blk_B_size/N_THREAD)*N_THREAD;// 0, 8, 16, 24...56 

    for(int k_cur=0; k_cur<K; k_cur+=K_BLOCK){
        int t_ld_off_in_blk_y = thread_ld_offset_in_blk_y;// 0~128

        // load A B block to slm
        #pragma unroll
        for(int t_ld_time = 0; t_ld_time<thread_ld_times; t_ld_time++){

            // load A blk
            FECTH_FLOAT4(&A_blk_tile[t_ld_off_in_blk_y][thread_ld_offset_in_blk_x*FLOAT4_NUMS])=
            FECTH_CONST_FLOAT4(A_blk_base_ptr+t_ld_off_in_blk_y*K+(thread_ld_offset_in_blk_x*FLOAT4_NUMS));// <<2 for we already know it is a float4
            
            // load B blk
            FECTH_FLOAT4(&B_blk_tile[t_ld_off_in_blk_y][thread_ld_offset_in_blk_x*FLOAT4_NUMS])=
            FECTH_CONST_FLOAT4(B_blk_base_ptr+t_ld_off_in_blk_y*K+(thread_ld_offset_in_blk_x*FLOAT4_NUMS));// <<2 for we already know it is a float4

            t_ld_off_in_blk_y+=warp_ld_cover_y_in_blk;
            
        }
        __syncthreads();

        // each warp accumlate C_thread_reg
        #pragma unroll
        for(int k_i=0; k_i<K_BLOCK; k_i++){
            // load slm to A_reg
            #pragma unroll
            for(int i=0; i<M_THREAD; i++){
                // A_thread_reg[i] = A_blk_tile[A_slm_warp_offset+A_reg_ld_x_idx+i][k_i];
                A_thread_reg[i] = A_blk_tile[A_slm_warp_offset+i][k_i];
            }
            
            // load slm to b_reg
            #pragma unroll
            for(int i=0; i<N_THREAD; i++){
                B_thread_reg[i] = B_blk_tile[B_slm_warp_offset+i][k_i];
            }

            // perform external product on A_reg x B_reg, store to C_reg
            #pragma unroll
            for(int i=0; i<M_THREAD; i++){
                #pragma unroll
                for(int j=0; j<N_THREAD; j++){
                    C_thread_reg[i][j] += A_thread_reg[i]*B_thread_reg[j];
                }
            }

        }
        __syncthreads();

        A_blk_base_ptr+=K_BLOCK;
        B_blk_base_ptr+=K_BLOCK;
    }

    // write C_reg to DRAM
    const int B_x_idx = blk_x*N_BLOCK+tx*N_THREAD;
    const int A_y_idx = blk_y*M_BLOCK+ty*M_THREAD;
    #pragma unroll
    for(int i=0; i<M_THREAD; i++){
        #pragma unroll
        for(int j=0; j<N_THREAD; j++){
            C[(A_y_idx+i)*N+B_x_idx+j] = C_thread_reg[i][j];
        }
    }

}

// + 71%
template <int M_BLOCK=128, int N_BLOCK=128, int K_BLOCK=8, int M_THREAD=8, int N_THREAD=8>
__global__ void sgemm_tiling_optimize(const float *A, const float *B, float *C, const size_t M, const size_t K, const size_t N)
{
    const int tx=threadIdx.x;
    const int ty=threadIdx.y;
    const int blk_x=blockIdx.x;
    const int blk_y=blockIdx.y;

    int warp_idx = (tx+ty*blockDim.x)/WARP_SIZE; // 0~8
    // 0~2
    int warp_lane = (tx+ty*blockDim.x)%WARP_SIZE; // 0~32
    // 0~32
    constexpr int warps_per_blk = 8; // 8
    // 2

    // for warps_blk_y ~ warps_blk_x covers most slm access latency
    constexpr int warps_blk_y = 2; // 2
    // 1: major direction
    constexpr int warps_blk_x = warps_per_blk / warps_blk_y; // 4
    // 2

    // For saving registers
    // assert(M_BLOCK==N_BLOCK) and (M_THREAD==N_THERAD)
    // so for loading A_blk and B_blk they share the same pattern
    constexpr int warp_k_ld_in_float4_per_row = K_BLOCK / FLOAT4_NUMS; // 8/4=2
    // 2
    const int A_thread_ld_offset_in_blk_y = M_BLOCK / warps_per_blk * warp_idx + warp_lane / warp_k_ld_in_float4_per_row; // (0~15)*8+0~15, t0 and t1 are the same
    const int B_thread_ld_offset_in_blk_x = M_BLOCK / warps_per_blk * warp_idx + warp_lane % (WARP_SIZE/2); // (0~15)*8+0~15, t0 and t16 are the same
    // 128*(0,1)+0~16
    const int A_thread_ld_offset_in_blk_x = warp_lane % warp_k_ld_in_float4_per_row;  // 0 for 0,2,4,...30 and 1 for 1,3,5,...31
    const int B_thread_ld_offset_in_blk_y = warp_lane / 16;  // 0 for 0,1,2,...15 and 1 for 16,17,18,..31
    // 0, 1
    constexpr int warp_ld_cover_y_in_blk = WARP_SIZE / warp_k_ld_in_float4_per_row; // 16
    // 16
    const int thread_ld_times = M_BLOCK / warps_per_blk / warp_ld_cover_y_in_blk; // 1
    // 8

    constexpr int slm_padding_dim = K_BLOCK;
    constexpr int slm_ty_padding_dim_per_row = N_BLOCK+FLOAT4_NUMS*2;
    constexpr int slm_ty_padding_dim = slm_ty_padding_dim_per_row*4+WARP_SIZE/2;
    __shared__ float A_blk_tile[slm_padding_dim][M_BLOCK];
    __shared__ float B_blk_tile[slm_padding_dim/4][slm_ty_padding_dim]; // avoid bank conflict, for SM Occupancy is limited by thread， +4 for padding for saving x,y,z,w, 4 for float4 loading

    const float* A_blk_base_ptr = A+blk_y*M_BLOCK*K;
    const float* B_blk_base_ptr = B+blk_x*N_BLOCK*K;

    float A_thread_reg [M_THREAD] ={0.f};
    float B_thread_reg [N_THREAD] = {0.f};
    float C_thread_reg [M_THREAD][N_THREAD] ={0.f};
    const int warp_C_offset_in_blk_row = warp_idx / warps_blk_y; // 0~4
    // 0~2
    const int warp_C_offset_in_blk_column = warp_idx % warps_blk_y; // 0~2
    // 0
    const int warps_blk_A_size = M_BLOCK/warps_blk_x; // 32
    const int warps_blk_B_size = N_BLOCK/warps_blk_y; // 64
    const int A_slm_warp_offset = ty*N_THREAD; // 0, 8, 16, 24, 32, 40, 48, ... 120(8*15)
    const int B_slm_warp_offset = tx*M_THREAD; // 0, 8, 16, 24, 32, 40, 48, ... 120(8*15)

    // const int A_reg_ld_x_idx = tx%(warps_blk_A_size/M_THREAD)*M_THREAD;// 0, 8, 16, 24
    // const int B_reg_ld_y_idx = tx%(warps_blk_B_size/N_THREAD)*N_THREAD;// 0, 8, 16, 24...56 
    A_blk_base_ptr += A_thread_ld_offset_in_blk_y*K;
    B_blk_base_ptr += B_thread_ld_offset_in_blk_x*K;

    const int B_x_idx = blk_x*N_BLOCK+tx*N_THREAD;
    const int A_y_idx = blk_y*M_BLOCK+ty*M_THREAD;

    const int ty_reg_offset_A = ty &1;

    const int tx_div_4 = tx >> 2;
    const int tx_mod_4 = tx & 0x3;
    const int tx_mod_8 = tx & 0x7;
    const int tx_addr_round_4 = warp_idx*16+(tx_div_4<<2);
    const int tx_mod_32 = tx_addr_round_4 & 0x1f;
    constexpr int tx_div_64_padding = 64+FLOAT4_NUMS;
    constexpr int tx_div_32_padding = 32+FLOAT4_NUMS;
    const int tx_div_64_offset = (tx_addr_round_4 >> 6) * tx_div_64_padding; // add 1 to avoid B t0 and t4 conflict
    const int tx_mod_8_equals_4_offset = ((tx_addr_round_4 & 0x7) == 4)*tx_div_32_padding;
    const int tx_mod_32_offset = (tx_mod_32>>3)<<3;
    const int tx_mod_64_div_32_offset = ((tx_addr_round_4 & 0x3f) >>5) <<2;

    const int B_no_conflict_store_offset=tx_mod_32_offset+tx_div_64_offset+tx_mod_8_equals_4_offset+tx_mod_64_div_32_offset+tx_mod_4;
    const int B_no_conflict_load_offset_first_half = ((tx >> 3) * tx_div_64_padding) + (tx_mod_4<<3) + ((tx_mod_8>>2)<<2);
    const int B_no_conflict_load_offset_second_half = B_no_conflict_load_offset_first_half+tx_div_32_padding;

    for(int k_cur=0; k_cur<K; k_cur+=K_BLOCK){

        // load A B block to slm
        {
            // load A blk in reg A
            FECTH_FLOAT4(A_thread_reg) = FECTH_CONST_FLOAT4(A_blk_base_ptr+(A_thread_ld_offset_in_blk_x*FLOAT4_NUMS));

            // transpose and store to A slm blk
            A_blk_tile[warp_idx][0*WARP_SIZE+(warp_lane>>1)+A_thread_ld_offset_in_blk_x*16] = reinterpret_cast<const float4*>(A_thread_reg)->x;
            A_blk_tile[warp_idx][1*WARP_SIZE+(warp_lane>>1)+A_thread_ld_offset_in_blk_x*16] = reinterpret_cast<const float4*>(A_thread_reg)->y;
            A_blk_tile[warp_idx][2*WARP_SIZE+(warp_lane>>1)+A_thread_ld_offset_in_blk_x*16] = reinterpret_cast<const float4*>(A_thread_reg)->z;
            A_blk_tile[warp_idx][3*WARP_SIZE+(warp_lane>>1)+A_thread_ld_offset_in_blk_x*16] = reinterpret_cast<const float4*>(A_thread_reg)->w;

            // load B blk in reg B
            FECTH_FLOAT4(B_thread_reg) = FECTH_CONST_FLOAT4(B_blk_base_ptr+(B_thread_ld_offset_in_blk_y*FLOAT4_NUMS));
            
            // transpose and store to B slm blk
            // if(B_thread_ld_offset_in_blk_y==0){

            B_blk_tile[B_thread_ld_offset_in_blk_y][B_no_conflict_store_offset+0*slm_ty_padding_dim_per_row] = reinterpret_cast<const float4*>(B_thread_reg)->x;
            B_blk_tile[B_thread_ld_offset_in_blk_y][B_no_conflict_store_offset+1*slm_ty_padding_dim_per_row] = reinterpret_cast<const float4*>(B_thread_reg)->y;
            B_blk_tile[B_thread_ld_offset_in_blk_y][B_no_conflict_store_offset+2*slm_ty_padding_dim_per_row] = reinterpret_cast<const float4*>(B_thread_reg)->z;
            B_blk_tile[B_thread_ld_offset_in_blk_y][B_no_conflict_store_offset+3*slm_ty_padding_dim_per_row] = reinterpret_cast<const float4*>(B_thread_reg)->w;
            // }

            // B_blk_tile[0+(B_thread_ld_offset_in_blk_y<<2)][B_no_conflict_store_offset] = reinterpret_cast<float4*>(B_thread_reg)->x;
            // B_blk_tile[1+(B_thread_ld_offset_in_blk_y<<2)][B_no_conflict_store_offset] = reinterpret_cast<float4*>(B_thread_reg)->y;
            // B_blk_tile[2+(B_thread_ld_offset_in_blk_y<<2)][B_no_conflict_store_offset] = reinterpret_cast<float4*>(B_thread_reg)->z;
            // B_blk_tile[3+(B_thread_ld_offset_in_blk_y<<2)][B_no_conflict_store_offset] = reinterpret_cast<float4*>(B_thread_reg)->w;

            // // load A blk
            // FECTH_FLOAT4(&A_blk_tile[thread_ld_offset_in_blk_y][thread_ld_offset_in_blk_x*FLOAT4_NUMS])=
            // FECTH_CONST_FLOAT4(A_blk_base_ptr+(thread_ld_offset_in_blk_x*FLOAT4_NUMS));// <<2 for we already know it is a float4
            
            // // load B blk
            // FECTH_FLOAT4(&B_blk_tile[thread_ld_offset_in_blk_y][thread_ld_offset_in_blk_x*FLOAT4_NUMS])=
            // FECTH_CONST_FLOAT4(B_blk_base_ptr+(thread_ld_offset_in_blk_x*FLOAT4_NUMS));// <<2 for we already know it is a float4   
        }
        __syncthreads();

        // each warp accumlate C_thread_reg
        #pragma unroll
        for(int k_i=0; k_i<K_BLOCK; k_i++){
            // load slm to A_reg
            int k_i_slm_blk_idx=k_i%4;
            int k_i_slm_blk_offset=k_i/4;

            // access slm in 128 bit to avoid bank conflict
            // A_thread_reg[i] = A_blk_tile[A_slm_warp_offset+i][k_i];

            // load no bank conflict for threads in a warp:
            // 0,2,4,...30 are accessing a same 4 float32(128 bit) in a row
            // 1,3,5,...31 are accessing a same 4 float32(128 bit) in a row
            // they all benifit from boardcast 
            FECTH_FLOAT4(A_thread_reg) = FECTH_CONST_FLOAT4(&A_blk_tile[warp_idx][k_i_slm_blk_idx*WARP_SIZE+k_i_slm_blk_offset*16+ty_reg_offset_A*8]);
            FECTH_FLOAT4(A_thread_reg+FLOAT4_NUMS) = FECTH_CONST_FLOAT4(&A_blk_tile[warp_idx][k_i_slm_blk_idx*WARP_SIZE+k_i_slm_blk_offset*16+ty_reg_offset_A*8+FLOAT4_NUMS]);

            
            // // load slm to b_reg
            // #pragma unroll
            // for(int i=0; i<N_THREAD; i++){
            //     B_thread_reg[i] = B_blk_tile[B_slm_warp_offset+i][k_i];
            // }

            // load slm to b_reg
            // t0 and t16 access same 8*fp32, it will be broadcast and so on
            const int k_i_mod_4 = k_i & 0x3;
            FECTH_FLOAT4(B_thread_reg) = FECTH_CONST_FLOAT4(&B_blk_tile[k_i>>2][B_no_conflict_load_offset_first_half+k_i_mod_4*slm_ty_padding_dim_per_row]);
            FECTH_FLOAT4(B_thread_reg+FLOAT4_NUMS) = FECTH_CONST_FLOAT4(&B_blk_tile[k_i>>2][B_no_conflict_load_offset_second_half+k_i_mod_4*slm_ty_padding_dim_per_row]);

            // #pragma unroll
            // for(int i=0; i<4; i++){
            //     B_thread_reg[i] = B_blk_tile[k_i][B_no_conflict_load_offset_first_half+i];
            // }
            // #pragma unroll
            // for(int i=0; i<4; i++){
            //     B_thread_reg[i+4] = B_blk_tile[k_i][B_no_conflict_load_offset_second_half+i];
            // }

            // perform external product on A_reg x B_reg, store to C_reg
            #pragma unroll
            for(int i=0; i<M_THREAD; i++){
                #pragma unroll
                for(int j=0; j<N_THREAD; j++){
                    C_thread_reg[i][j] += A_thread_reg[i]*B_thread_reg[j];
                }
            }

        }
        __syncthreads();

        A_blk_base_ptr+=K_BLOCK;
        B_blk_base_ptr+=K_BLOCK;
    }

    // write C_reg to DRAM

    #pragma unroll
    for(int i=0; i<M_THREAD; i++){
        FECTH_FLOAT4(&C[(A_y_idx+i)*N+B_x_idx+0]) = FECTH_CONST_FLOAT4(&C_thread_reg[i][0]);
        FECTH_FLOAT4(&C[(A_y_idx+i)*N+B_x_idx+FLOAT4_NUMS]) = FECTH_CONST_FLOAT4(&C_thread_reg[i][FLOAT4_NUMS]);
        // #pragma unroll
        // for(int j=0; j<N_THREAD; j++){
        //     C[(A_y_idx+i)*N+B_x_idx+j] = C_thread_reg[i][j];
        // }
    }

}

// +2%
template <int M_BLOCK=128, int N_BLOCK=128, int K_BLOCK=8, int M_THREAD=8, int N_THREAD=8, bool CROSS_LOAD_B=true>
__global__ void sgemm_tiling_warp_optimize(const float *A, const float *B, float *C, const size_t M, const size_t K, const size_t N)
{
    const int tx=threadIdx.x;
    const int ty=threadIdx.y;
    const int blk_x=blockIdx.x;
    const int blk_y=blockIdx.y;

    int warp_idx = (tx+ty*blockDim.x)/WARP_SIZE; // 0~8
    // 0~2
    int warp_lane = (tx+ty*blockDim.x)%WARP_SIZE; // 0~32
    // 0~32
    constexpr int warps_per_blk = 8; // 8
    // 2

    // for warps_blk_y ~ warps_blk_x covers most slm access latency
    constexpr int warps_blk_y = 2; // 2
    // 1: major direction
    constexpr int warps_blk_x = warps_per_blk / warps_blk_y; // 4
    // 2

    // For saving registers
    // assert(M_BLOCK==N_BLOCK) and (M_THREAD==N_THERAD)
    // so for loading A_blk and B_blk they share the same pattern
    constexpr int warp_k_ld_in_float4_per_row = K_BLOCK / FLOAT4_NUMS; // 8/4=2
    // 2
    const int A_thread_ld_offset_in_blk_y = M_BLOCK / warps_per_blk * warp_idx + warp_lane / warp_k_ld_in_float4_per_row; // (0~15)*8+0~15, t0 and t1 are the same
    const int B_thread_ld_offset_in_blk_x = M_BLOCK / warps_per_blk * warp_idx + warp_lane % (WARP_SIZE/2); // (0~15)*8+0~15, t0 and t16 are the same
    // 128*(0,1)+0~16
    const int A_thread_ld_offset_in_blk_x = warp_lane % warp_k_ld_in_float4_per_row;  // 0 for 0,2,4,...30 and 1 for 1,3,5,...31
    const int B_thread_ld_offset_in_blk_y = warp_lane / 16;  // 0 for 0,1,2,...15 and 1 for 16,17,18,..31
    // 0, 1
    constexpr int warp_ld_cover_y_in_blk = WARP_SIZE / warp_k_ld_in_float4_per_row; // 16
    // 16
    const int thread_ld_times = M_BLOCK / warps_per_blk / warp_ld_cover_y_in_blk; // 1
    // 8

    constexpr int slm_padding_dim = K_BLOCK;
    // constexpr int slm_ty_padding_dim_per_row = N_BLOCK+FLOAT4_NUMS*2;
    constexpr int slm_ty_padding_dim = N_BLOCK+FLOAT4_NUMS;
    __shared__ float A_blk_tile[slm_padding_dim][M_BLOCK];
    __shared__ float B_blk_tile[slm_padding_dim][slm_ty_padding_dim]; // avoid bank conflict, for SM Occupancy is limited by thread， +4 for padding for saving x,y,z,w, 4 for float4 loading

    const float* A_blk_base_ptr = A+blk_y*M_BLOCK*K;
    const float* B_blk_base_ptr = B+blk_x*N_BLOCK*K;

    float A_thread_reg [M_THREAD] ={0.f};
    float B_thread_reg [N_THREAD] = {0.f};
    float C_thread_reg [M_THREAD][N_THREAD] ={0.f};
    const int warp_C_offset_in_blk_row = warp_idx / warps_blk_y; // 0~4
    // 0~2
    const int warp_C_offset_in_blk_column = warp_idx % warps_blk_y; // 0~2
    // 0
    const int warps_blk_A_size = M_BLOCK/warps_blk_x; // 32
    const int warps_blk_B_size = N_BLOCK/warps_blk_y; // 64
    const int A_slm_warp_offset = ty*N_THREAD; // 0, 8, 16, 24, 32, 40, 48, ... 120(8*15)
    const int B_slm_warp_offset = tx*M_THREAD; // 0, 8, 16, 24, 32, 40, 48, ... 120(8*15)

    A_blk_base_ptr += A_thread_ld_offset_in_blk_y*K;
    B_blk_base_ptr += B_thread_ld_offset_in_blk_x*K;

    
    const int ty_reg_offset_A = ty &1;
    
    const int tx_div_4 = tx >> 2;
    const int tx_mod_4 = tx & 0x3;
    const int tx_mod_8 = tx & 0x7;
    
    const int warp_idx_div_2 = warp_idx >> 1;
    const int warp_idx_mod_2 = warp_idx % 2;
    const int lane_div_16 = warp_lane >> 4; // 0 for 0~15, 1 for 16~31
    const int lane_mod_16 = warp_lane % 16;
    const int lane_mod_2 = warp_lane & 1;
    const int lane_mod_16_div2 = (warp_lane & 0xf) >> 1;

    const int warp_quarter_idx_mod_2 = (warp_lane >> 3)%2;
    
    const int A_y_idx = blk_y*M_BLOCK+warp_idx_div_2*WARP_SIZE+lane_div_16*16+lane_mod_2*8;
    const int B_x_idx = blk_x*N_BLOCK+warp_idx_mod_2*64+lane_mod_16_div2*8;

    for(int k_cur=0; k_cur<K; k_cur+=K_BLOCK){

        // load A B block to slm
        {
            // load A blk in reg A
            FECTH_FLOAT4(A_thread_reg) = FECTH_CONST_FLOAT4(A_blk_base_ptr+(A_thread_ld_offset_in_blk_x*FLOAT4_NUMS));

            // transpose and store to A slm blk
            A_blk_tile[warp_idx][0*WARP_SIZE+(warp_lane>>1)+A_thread_ld_offset_in_blk_x*16] = reinterpret_cast<const float4*>(A_thread_reg)->x;
            A_blk_tile[warp_idx][1*WARP_SIZE+(warp_lane>>1)+A_thread_ld_offset_in_blk_x*16] = reinterpret_cast<const float4*>(A_thread_reg)->y;
            A_blk_tile[warp_idx][2*WARP_SIZE+(warp_lane>>1)+A_thread_ld_offset_in_blk_x*16] = reinterpret_cast<const float4*>(A_thread_reg)->z;
            A_blk_tile[warp_idx][3*WARP_SIZE+(warp_lane>>1)+A_thread_ld_offset_in_blk_x*16] = reinterpret_cast<const float4*>(A_thread_reg)->w;

            // load B blk in reg B
            FECTH_FLOAT4(B_thread_reg) = FECTH_CONST_FLOAT4(B_blk_base_ptr+(B_thread_ld_offset_in_blk_y*FLOAT4_NUMS));
            
            // transpose and store to B slm blk

            B_blk_tile[(B_thread_ld_offset_in_blk_y<<2)+0][warp_idx*16+lane_mod_16] = reinterpret_cast<const float4*>(B_thread_reg)->x;
            B_blk_tile[(B_thread_ld_offset_in_blk_y<<2)+1][warp_idx*16+lane_mod_16] = reinterpret_cast<const float4*>(B_thread_reg)->y;
            B_blk_tile[(B_thread_ld_offset_in_blk_y<<2)+2][warp_idx*16+lane_mod_16] = reinterpret_cast<const float4*>(B_thread_reg)->z;
            B_blk_tile[(B_thread_ld_offset_in_blk_y<<2)+3][warp_idx*16+lane_mod_16] = reinterpret_cast<const float4*>(B_thread_reg)->w;
        }
        __syncthreads();

        // each warp accumlate C_thread_reg
        // in a warp the thread map looks like:
        // 0,2,4,6,8,10,12,14
        // 1,3,5,7,9,11,13,15
        // 16, 18, 20, 22, 24, 26, 28, 30
        // 17, 19, 21, 23, 25, 27, 29, 31
        #pragma unroll
        for(int k_i=0; k_i<K_BLOCK; k_i++){
            // load slm to A_reg
            int k_i_slm_blk_idx=k_i%4;
            int k_i_slm_blk_offset=k_i/4;

            // load no bank conflict for threads in a warp:
            // 0,2,4,6,8,10,12,14 are accessing a same 8*fp32 in a row
            // 1,3,5,7,9,11,13,15 are accessing a same 8*fp32 in a row
            // 16, 18, 20, 22, 24, 26, 28, 30 are accessing a same 8*fp32 in a row
            // 17, 19, 21, 23, 25, 27, 29, 31 are accessing a same 8*fp32 in a row
            // thread in a row benifits from broadcast
            FECTH_FLOAT4(A_thread_reg) = FECTH_CONST_FLOAT4(&A_blk_tile[warp_idx_div_2*2+lane_div_16][(k_i_slm_blk_idx)*32+k_i_slm_blk_offset*16+lane_mod_2*8]);
            FECTH_FLOAT4(A_thread_reg+FLOAT4_NUMS) = FECTH_CONST_FLOAT4(&A_blk_tile[warp_idx_div_2*2+lane_div_16][(k_i_slm_blk_idx)*32+k_i_slm_blk_offset*16+lane_mod_2*8+FLOAT4_NUMS]);

            // load no bank conflict for threads in a warp:
            // 0,1,16,17 are accessing a same 8*fp32 in a column
            // 2,3,18,19 are accessing a same 8*fp32 in a column
            // 4,5,20,21 are accessing a same 8*fp32 in a column
            // ...
            // 14,15,30,31 are accessing a same 8*fp32 in a column
            const int k_i_mod_4 = k_i & 0x3;
            // t0 t8 bank conflict, let first quarter and second quarter access different float4
            if(CROSS_LOAD_B)
            {
                FECTH_FLOAT4(B_thread_reg+FLOAT4_NUMS*warp_quarter_idx_mod_2) = FECTH_CONST_FLOAT4(&B_blk_tile[k_i][warp_idx_mod_2*64+lane_mod_16_div2*8+FLOAT4_NUMS*warp_quarter_idx_mod_2]);
                FECTH_FLOAT4(B_thread_reg+FLOAT4_NUMS*(warp_quarter_idx_mod_2^1)) = FECTH_CONST_FLOAT4(&B_blk_tile[k_i][warp_idx_mod_2*64+lane_mod_16_div2*8+FLOAT4_NUMS*(warp_quarter_idx_mod_2^1)]);
                // FECTH_FLOAT4(B_thread_reg+FLOAT4_NUMS) = FECTH_CONST_FLOAT4(&B_blk_tile[k_i][warp_idx_mod_2*64+lane_mod_16_div2*8+FLOAT4_NUMS]);
            }else
            // cross load version, t0 and t8 access different parts of 8*fp32 by float4 step
            // notice how we use warp_quarter_idx_mod_2
            {
                FECTH_FLOAT4(B_thread_reg) = FECTH_CONST_FLOAT4(&B_blk_tile[k_i][warp_idx_mod_2*64+lane_mod_16_div2*8]);
                FECTH_FLOAT4(B_thread_reg+FLOAT4_NUMS) = FECTH_CONST_FLOAT4(&B_blk_tile[k_i][warp_idx_mod_2*64+lane_mod_16_div2*8+FLOAT4_NUMS]);
            }
 

            // perform external product on A_reg x B_reg, store to C_reg
            #pragma unroll
            for(int i=0; i<M_THREAD; i++){
                #pragma unroll
                for(int j=0; j<N_THREAD; j++){
                    C_thread_reg[i][j] += A_thread_reg[i]*B_thread_reg[j];
                }
            }

        }
        __syncthreads();

        A_blk_base_ptr+=K_BLOCK;
        B_blk_base_ptr+=K_BLOCK;
    }

    // write C_reg to DRAM

    #pragma unroll
    for(int i=0; i<M_THREAD; i++){
        FECTH_FLOAT4(&C[(A_y_idx+i)*N+B_x_idx+0]) = FECTH_CONST_FLOAT4(&C_thread_reg[i][0]);
        FECTH_FLOAT4(&C[(A_y_idx+i)*N+B_x_idx+FLOAT4_NUMS]) = FECTH_CONST_FLOAT4(&C_thread_reg[i][FLOAT4_NUMS]);
    }

}



template <typename T, int SL = 1024, int HA = 4096, int HB = 4096>
void test_with_dtype(qttbench::State &state)
{
    constexpr int bs = 1, seq_len = SL, hidden_status_A = HA, hidden_status_B = HB;

    qttbench::Tensor<qttbench::float32_t> A(3, {hidden_status_A, seq_len, bs});
    qttbench::Tensor<qttbench::float32_t> B(3, {hidden_status_A, hidden_status_B, bs});
    A.initialize_random();
    B.initialize_random();

    // A.print();
    // B.print();

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

    // state.run(
    //     "sgemm_native",
    //     [&](cudaStream_t s)
    //     {
    //         const size_t K = hidden_status_A;
    //         const size_t N = hidden_status_B;
    //         const size_t M = seq_len;
    //         const size_t ne2 = bs;
    //         constexpr int BLOCK_SIZE = 256;

    //         dim3 grid_size((N + BLOCK_SIZE - 1) / BLOCK_SIZE, M, ne2);
    //         dim3 block_size(BLOCK_SIZE, 1, 1);
    //         sgemm_native<BLOCK_SIZE><<<grid_size, block_size, 0, s>>>(
    //             A.data_ptr(), B.data_ptr(), C.data_ptr(), M, K, N);
    //     },
    //     VERIFY_FUNC);

    state.run(
        "sgemm_tiling 128 128 8 8 8",
        [&](cudaStream_t s)
        {
            const size_t K = hidden_status_A;
            const size_t N = hidden_status_B;
            const size_t M = seq_len;
            const size_t ne2 = bs;
            constexpr int BLOCK_SIZE = 128;
            constexpr int THREAD_SIZE = 8;
            constexpr int K_BLOCK_SIZE = 8;

            dim3 grid_size((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M+BLOCK_SIZE-1)/ BLOCK_SIZE, ne2);
            dim3 block_size(BLOCK_SIZE/THREAD_SIZE, BLOCK_SIZE/THREAD_SIZE, 1);
            // cudaFuncSetAttribute(sgemm_tiling_base<BLOCK_SIZE, BLOCK_SIZE, K_BLOCK_SIZE, THREAD_SIZE, THREAD_SIZE>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
            sgemm_tiling_base<BLOCK_SIZE, BLOCK_SIZE, K_BLOCK_SIZE, THREAD_SIZE, THREAD_SIZE><<<grid_size, block_size, 0, s>>>(
                A.data_ptr(), B.data_ptr(), C.data_ptr(), M, K, N);
        },
        VERIFY_FUNC);

        state.run(
        "sgemm_tiling optimize 128 128 8 8 8",
        [&](cudaStream_t s)
        {
            const size_t K = hidden_status_A;
            const size_t N = hidden_status_B;
            const size_t M = seq_len;
            const size_t ne2 = bs;
            constexpr int BLOCK_SIZE = 128;
            constexpr int THREAD_SIZE = 8;
            constexpr int K_BLOCK_SIZE = 8;

            dim3 grid_size((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M+BLOCK_SIZE-1)/ BLOCK_SIZE, ne2);
            dim3 block_size(BLOCK_SIZE/THREAD_SIZE, BLOCK_SIZE/THREAD_SIZE, 1);
            // cudaFuncSetAttribute(sgemm_tiling_base<BLOCK_SIZE, BLOCK_SIZE, K_BLOCK_SIZE, THREAD_SIZE, THREAD_SIZE>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
            sgemm_tiling_optimize<BLOCK_SIZE, BLOCK_SIZE, K_BLOCK_SIZE, THREAD_SIZE, THREAD_SIZE><<<grid_size, block_size, 0, s>>>(
                A.data_ptr(), B.data_ptr(), C.data_ptr(), M, K, N);
        },
        VERIFY_FUNC);

        state.run(
        "sgemm_tiling optimize warp 128 128 8 8 8",
        [&](cudaStream_t s)
        {
            const size_t K = hidden_status_A;
            const size_t N = hidden_status_B;
            const size_t M = seq_len;
            const size_t ne2 = bs;
            constexpr int BLOCK_SIZE = 128;
            constexpr int THREAD_SIZE = 8;
            constexpr int K_BLOCK_SIZE = 8;

            dim3 grid_size((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M+BLOCK_SIZE-1)/ BLOCK_SIZE, ne2);
            dim3 block_size(BLOCK_SIZE/THREAD_SIZE, BLOCK_SIZE/THREAD_SIZE, 1);
            // cudaFuncSetAttribute(sgemm_tiling_base<BLOCK_SIZE, BLOCK_SIZE, K_BLOCK_SIZE, THREAD_SIZE, THREAD_SIZE>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
            sgemm_tiling_warp_optimize<BLOCK_SIZE, BLOCK_SIZE, K_BLOCK_SIZE, THREAD_SIZE, THREAD_SIZE, false><<<grid_size, block_size, 0, s>>>(
                A.data_ptr(), B.data_ptr(), C.data_ptr(), M, K, N);
        },
        VERIFY_FUNC);

        state.run(
        "sgemm_tiling optimize warp cross load B 128 128 8 8 8",
        [&](cudaStream_t s)
        {
            const size_t K = hidden_status_A;
            const size_t N = hidden_status_B;
            const size_t M = seq_len;
            const size_t ne2 = bs;
            constexpr int BLOCK_SIZE = 128;
            constexpr int THREAD_SIZE = 8;
            constexpr int K_BLOCK_SIZE = 8;

            dim3 grid_size((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M+BLOCK_SIZE-1)/ BLOCK_SIZE, ne2);
            dim3 block_size(BLOCK_SIZE/THREAD_SIZE, BLOCK_SIZE/THREAD_SIZE, 1);
            // cudaFuncSetAttribute(sgemm_tiling_base<BLOCK_SIZE, BLOCK_SIZE, K_BLOCK_SIZE, THREAD_SIZE, THREAD_SIZE>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
            sgemm_tiling_warp_optimize<BLOCK_SIZE, BLOCK_SIZE, K_BLOCK_SIZE, THREAD_SIZE, THREAD_SIZE><<<grid_size, block_size, 0, s>>>(
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

    // test_with_dtype<qttbench::float32_t, 128, 128, 128>(state);
    test_with_dtype<qttbench::float32_t, 4096, 4096, 4096>(state);
    // test_with_dtype<qttbench::float32_t, 2, 4096, 14336>(state);
    // test_with_dtype<qttbench::float32_t, 4096, 14336>(state);
    return 0;
}
