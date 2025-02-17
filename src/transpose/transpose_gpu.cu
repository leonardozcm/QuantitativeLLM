#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "qttbench/qtt_state.h"

using namespace std;

template <typename T>
__global__ void transpose_gpu_native(const T *matrix, T *m_t, int M, int N)
{
    int b_x = blockIdx.x;
    int b_y = blockIdx.y;
    int t_x = threadIdx.x;
    int t_y = threadIdx.y;

    int x_id = b_x * blockDim.x + t_x;
    int y_id = b_y * blockDim.y + t_y;
    m_t[x_id * N + y_id] = matrix[y_id * N + x_id];
}

template <typename T>
__global__ void transpose_gpu_pretranspose_float4(const T *matrix, T *m_t, int M, int N){
    int col_idx=blockIdx.x*blockDim.x+threadIdx.x;
    int row_idx=blockIdx.y*blockDim.y+threadIdx.y;

    int offset=(row_idx*N+col_idx)<<2;
    const float4* v_t = reinterpret_cast<const float4*>(matrix+offset);

    float4 v_0 = v_t[0];
    float4 v_1 = v_t[N>>2];
    float4 v_2 = v_t[N>>1];
    float4 v_3 = v_t[(N>>2)*3];

    float4 v_t_0 = make_float4(v_0.x, v_1.x, v_2.x, v_3.x);
    float4 v_t_1 = make_float4(v_0.y, v_1.y, v_2.y, v_3.y);
    float4 v_t_2 = make_float4(v_0.z, v_1.z, v_2.z, v_3.z);
    float4 v_t_3 = make_float4(v_0.w, v_1.w, v_2.w, v_3.w);

    // write back
    offset=(col_idx*N+row_idx)<<2;
    float4* v_tt = reinterpret_cast<float4*>(m_t+offset);
    v_tt[0] = v_t_0;
    v_tt[N>>2] = v_t_1;
    v_tt[N>>1] = v_t_2;
    v_tt[(N>>2)*3] = v_t_3;

}

template <typename T>
__global__ void transpose_gpu_pretranspose_float2(const T *matrix, T *m_t, int M, int N){
    int col_idx=blockIdx.x*blockDim.x+threadIdx.x;
    int row_idx=blockIdx.y*blockDim.y+threadIdx.y;

    int offset=(row_idx*N+col_idx)<<1;
    const float2* v_t = reinterpret_cast<const float2*>(matrix+offset);

    float2 v_0 = v_t[0];
    float2 v_1 = v_t[N>>1];

    float2 v_t_0 = make_float2(v_0.x, v_1.x);
    float2 v_t_1 = make_float2(v_0.y, v_1.y);

    // write back
    offset=(col_idx*N+row_idx)<<1;
    float2* v_tt = reinterpret_cast<float2*>(m_t+offset);
    v_tt[0] = v_t_0;
    v_tt[N>>1] = v_t_1;

}

template <typename T>
__global__ void transpose_gpu_pretranspose_float2row1(const T *matrix, T *m_t, int M, int N){
    int col_idx=blockIdx.x*blockDim.x+threadIdx.x;
    int row_idx=blockIdx.y*blockDim.y+threadIdx.y;

    int offset= (row_idx<<1)*N  + col_idx;
    const float* v_t = matrix+offset;

    float2 v_tranposed = make_float2(v_t[0], v_t[N]);

    // write back
    offset=(col_idx*N) + (row_idx<<1);
    float2* v_tt = reinterpret_cast<float2*>(m_t+offset);
    *(v_tt)=v_tranposed;

}


// Deprecated
//
// template <typename T>
// __global__ void transpose_gpu_ldg_read(const T *matrix, T *m_t, int M, int N)
// {
//     int b_x = blockIdx.x;
//     int b_y = blockIdx.y;
//     int t_x = threadIdx.x;
//     int t_y = threadIdx.y;

//     int x_id = b_x * blockDim.x + t_x;
//     int y_id = b_y * blockDim.y + t_y;
//     m_t[x_id * N + y_id] = __ldg(&matrix[y_id * N + x_id]);
// }

template <typename T>
__global__ void transpose_gpu_merged_write(const T *matrix, T *m_t, int M, int N)
{
    int b_x = blockIdx.x;
    int b_y = blockIdx.y;
    int t_x = threadIdx.x;
    int t_y = threadIdx.y;

    int x_id = b_x * blockDim.x + t_x;
    int y_id = b_y * blockDim.y + t_y;
    m_t[y_id * N + x_id] = matrix[x_id * N + y_id];
}

// Deprecated
//
// template <typename T>
// __global__ void transpose_gpu_merged_write_ldg_read(const T *matrix, T *m_t, int M, int N)
// {
//     int b_x = blockIdx.x;
//     int b_y = blockIdx.y;
//     int t_x = threadIdx.x;
//     int t_y = threadIdx.y;

//     int x_id = b_x * blockDim.x + t_x;
//     int y_id = b_y * blockDim.y + t_y;
//     m_t[y_id * N + x_id] = __ldg(&matrix[x_id * N + y_id]);
// }

// Shared Memory
template <typename T, int TILE_SIZE>
__global__ void transpose_gpu_shared(const T *matrix, T *m_t, int M, int N)
{
    __shared__ float slm_m[TILE_SIZE][TILE_SIZE]; // assert TILE_SIZE == grid_size
    const int b_x = blockIdx.x;
    const int b_y = blockIdx.y;
    const int t_x = threadIdx.x;
    const int t_y = threadIdx.y;
    const int x_id = b_x * blockDim.x + t_x;
    const int y_id = b_y * blockDim.y + t_y;

    slm_m[t_y][t_x] = matrix[y_id * N + x_id];
    __syncthreads();

    // 8-way bank conflict if TILE_SIZE==16:
    // t_x0                     t_x16
    // [0],   1,  2,  3, ... 15, 16,  17, 18, 19, ... 31
    // [32], 33, 34, 35, ... 47, 48
    // ...
    // [32 * 16 / 2]
    //
    // 32-way bank conflict if TILE_SIZE==32
    // t_x0
    // [0 ],  1,  2,  3, ... 31
    // [32], 33, 34, 35, ... 63
    // ...
    // [32 * 32]
    //
    
    const int x2 = b_y * blockDim.x + t_x;
    const int y2 = b_x * blockDim.y + t_y;
    m_t[y2 * N + x2] = slm_m[t_x][t_y];
}

// Shared Memory
template <typename T, int TILE_SIZE>
__global__ void transpose_gpu_shared_bank_conflict_optimized(const T *matrix, T *m_t, int M, int N)
{
    __shared__ float slm_m[TILE_SIZE][TILE_SIZE + 1]; // assert TILE_SIZE == grid_size
    const int b_x = blockIdx.x;
    const int b_y = blockIdx.y;
    const int t_x = threadIdx.x;
    const int t_y = threadIdx.y;
    const int x_id = b_x * blockDim.x + t_x;
    const int y_id = b_y * blockDim.y + t_y;

    slm_m[t_y][t_x] = matrix[y_id * N + x_id];
    __syncthreads();
    // (N * 17) % 32, N= 0, 1, 2, ..15
    const int x2 = b_y * blockDim.x + t_x;
    const int y2 = b_x * blockDim.y + t_y;
    m_t[y2 * N + x2] = slm_m[t_x][t_y];
}

template <typename T>
__global__ void reset_transpose_gpu(T *m_t, int M, int N)
{
    int b_x = blockIdx.x;
    int b_y = blockIdx.y;
    int t_x = threadIdx.x;
    int t_y = threadIdx.y;

    int x_id = b_x * blockDim.x + t_x;
    int y_id = b_y * blockDim.y + t_y;
    m_t[y_id * N + x_id] = 0.f;
}

template <typename T>
__global__ void verify_transpose_gpu(const T *matrix, T *m_t, int M, int N, int *ret)
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
    if ((*ret) && matrix[x_id * N + y_id] != m_t[y_id * N + x_id])
    {
        (*ret) = 0;
    }
}

#define VERIFY_FUNC                                                                                \
    [&](cudaStream_t s)                                                                            \
    {                                                                                              \
        dim3 grid_size(N / 16, N / 16);                                                           \
        dim3 block_size(16, 16);                                                                  \
        int *d_correct;                                                                            \
        int correct = 1;                                                                           \
        cudaMalloc(&d_correct, sizeof(int));                                                       \
        cudaMemcpy(d_correct, &correct, sizeof(int), cudaMemcpyHostToDevice);                      \
        verify_transpose_gpu<<<grid_size, block_size, 0, s>>>(d_matrix, d_m_t, N, N, d_correct); \
        cudaMemcpy(&correct, d_correct, sizeof(int), cudaMemcpyDeviceToHost);                      \
        reset_transpose_gpu<<<grid_size, block_size, 0, s>>>(d_m_t, N, N);                       \
        return correct;                                                                            \
    }

template <typename T>
void test_with_dtype(qttbench::State &state)
{
    constexpr size_t N = 1024 * 16;
    T *matrix = (T *)malloc(N * N * sizeof(T));
    T *m_t = (T *)malloc(N * N * sizeof(T));

    // Init matrix
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            matrix[i * N + j] = i * N + j;
        }
    }

    T *d_matrix;
    T *d_m_t;
    size_t size = N * N * sizeof(T);
    cudaMalloc(&d_matrix, size);
    cudaMalloc(&d_m_t, size);

    cudaMemcpy(d_matrix, matrix, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_m_t, m_t, size, cudaMemcpyHostToDevice);

    state.run(
        "transpose_gpu native kernel",
        [&](cudaStream_t s)
        {
            constexpr int block_width = 16;
            dim3 grid_size(N / block_width, N / block_width);
            dim3 block_size(block_width, block_width);
            transpose_gpu_native<<<grid_size, block_size, 0, s>>>(d_matrix, d_m_t, N, N);
        },
        VERIFY_FUNC);

    state.run(
        "transpose_gpu_pretranspose(4x4)",
        [&](cudaStream_t s)
        {
            constexpr int block_width = 16;
            constexpr int register_blk_size = 4;
            dim3 grid_size(N / block_width / register_blk_size, N / block_width / register_blk_size);
            dim3 block_size(block_width, block_width);
            transpose_gpu_pretranspose_float4<<<grid_size, block_size, 0, s>>>(d_matrix, d_m_t, N, N);
        },
        VERIFY_FUNC);

    state.run(
        "transpose_gpu_pretranspose(2x2)",
        [&](cudaStream_t s)
        {
            constexpr int block_height = 32;
            constexpr int block_width = 8;
            constexpr int register_blk_size = 2;
            dim3 grid_size(N / block_width / register_blk_size, N / block_height / register_blk_size);
            dim3 block_size(block_width, block_height);
            transpose_gpu_pretranspose_float2<<<grid_size, block_size, 0, s>>>(d_matrix, d_m_t, N, N);
        },
        VERIFY_FUNC);

    state.run(
        "transpose_gpu_pretranspose(2x1)",
        [&](cudaStream_t s)
        {
            constexpr int block_height = 32;
            constexpr int block_width = 8;
            constexpr int register_blk_size = 2;
            dim3 grid_size(N / block_width , N / block_height / register_blk_size);
            dim3 block_size(block_width, block_height);
            transpose_gpu_pretranspose_float2row1<<<grid_size, block_size, 0, s>>>(d_matrix, d_m_t, N, N);
        },
        VERIFY_FUNC);

    state.run(
        "transpose_gpu native kernel BLOCK SIZE 8",
        [&](cudaStream_t s)
        {
            constexpr int block_width = 8;
            dim3 grid_size(N / block_width, N / block_width);
            dim3 block_size(block_width, block_width);
            transpose_gpu_native<<<grid_size, block_size, 0, s>>>(d_matrix, d_m_t, N, N);
        },
        VERIFY_FUNC);

    state.run(
        "transpose_gpu native kernel BLOCK SIZE 32",
        [&](cudaStream_t s)
        {
            constexpr int block_width = 32;
            dim3 grid_size(N / block_width, N / block_width);
            dim3 block_size(block_width, block_width);
            transpose_gpu_native<<<grid_size, block_size, 0, s>>>(d_matrix, d_m_t, N, N);
        },
        VERIFY_FUNC);

    // state.run(
    //     "transpose_gpu ldg read kernel",
    //     [&](cudaStream_t s)
    //     {
    //         constexpr int block_width = 16;
    //         dim3 grid_size(N / block_width, N / block_width);
    //         dim3 block_size(block_width, block_width);
    //         transpose_gpu_ldg_read<<<grid_size, block_size, 0, s>>>(d_matrix, d_m_t, N, N);
    //     },
    //     VERIFY_FUNC);

    state.run(
        "transpose_gpu merged write kernel",
        [&](cudaStream_t s)
        {
            constexpr int block_width = 16;
            dim3 grid_size(N / block_width, N / block_width);
            dim3 block_size(block_width, block_width);
            transpose_gpu_merged_write<<<grid_size, block_size, 0, s>>>(d_matrix, d_m_t, N, N);
        },
        VERIFY_FUNC);

    // state.run(
    //     "transpose_gpu merged write _ldg read kernel",
    //     [&](cudaStream_t s)
    //     {
    //         constexpr int block_width = 16;
    //         dim3 grid_size(N / block_width, N / block_width);
    //         dim3 block_size(block_width, block_width);
    //         transpose_gpu_merged_write_ldg_read<<<grid_size, block_size, 0, s>>>(d_matrix, d_m_t, N, N);
    //     },
    //     VERIFY_FUNC);

    state.run(
        "transpose_gpu slm kernel",
        [&](cudaStream_t s)
        {
            constexpr int block_width = 16;
            dim3 grid_size(N / block_width, N / block_width);
            dim3 block_size(block_width, block_width);
            transpose_gpu_shared<T, block_width><<<grid_size, block_size, 0, s>>>(d_matrix, d_m_t, N, N);
        },
        VERIFY_FUNC);

    state.run(
        "transpose_gpu slm kernel BLOCK SIZE 32",
        [&](cudaStream_t s)
        {
            constexpr int block_width = 32;
            dim3 grid_size(N / block_width, N / block_width);
            dim3 block_size(block_width, block_width);
            transpose_gpu_shared<T, block_width><<<grid_size, block_size, 0, s>>>(d_matrix, d_m_t, N, N);
        },
        VERIFY_FUNC);

    state.run(
        "transpose_gpu slm padding kernel",
        [&](cudaStream_t s)
        {
            constexpr int block_width = 16;
            dim3 grid_size(N / block_width, N / block_width);
            dim3 block_size(block_width, block_width);
            transpose_gpu_shared_bank_conflict_optimized<T, block_width><<<grid_size, block_size, 0, s>>>(d_matrix, d_m_t, N, N);
        },
        VERIFY_FUNC);

    state.run(
        "transpose_gpu slm padding kernel BLOCK SIZE 32 ",
        [&](cudaStream_t s)
        {
            constexpr int block_width = 32;
            dim3 grid_size(N / block_width, N / block_width);
            dim3 block_size(block_width, block_width);
            transpose_gpu_shared_bank_conflict_optimized<T, block_width><<<grid_size, block_size, 0, s>>>(d_matrix, d_m_t, N, N);
        },
        VERIFY_FUNC);

    cudaFree(d_matrix);
    cudaFree(d_m_t);

    free(matrix);
    free(m_t);

    state.dump_csv();
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
