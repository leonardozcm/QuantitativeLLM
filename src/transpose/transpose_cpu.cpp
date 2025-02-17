#include <chrono>
#include <iostream>
#include <string>
#include <functional>
#include <stdio.h>
#include "utils.h"
#include "cpu_arch/i9_13900K_profile.h"

#include "qttbench/qtt_state.h"

using namespace std;

template <typename T>
void clear(T *m, int M, int N)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            m[i * N + j] = 0;
        }
    }
}

template <typename T>
void transposeNative(const T *matrix, T *m_t, int M, int N)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            m_t[j * N + i] = matrix[i * N + j];
        }
    }
}

template <typename T, size_t CS>
void chunkedTranspose(const T *matrix, T *m_t, int M, int N)
{
    const size_t M_CHUNK = M / CS;
    const size_t N_CHUNK = N / CS;
    for (int i = 0; i < M_CHUNK; i++)
    {
        for (int j = 0; j < N_CHUNK; j++)
        {
            size_t i_offset = i * CS;
            size_t j_offset = j * CS;
            for (int ii = 0; ii < CS; ii++)
            {
#pragma unroll
                for (int jj = 0; jj < CS; jj++)
                {
                    m_t[(j_offset + jj) * N + i_offset + ii] = matrix[(i_offset + ii) * N + j_offset + jj];
                }
            }
        }
    }
}

template <typename T>
void chunkedTranspose_unloop_8(const T *matrix, T *m_t, int M, int N)
{
    const size_t M_CHUNK = M / 8;
    const size_t N_CHUNK = N / 8;
    for (int i = 0; i < M_CHUNK; i++)
    {
        for (int j = 0; j < N_CHUNK; j++)
        {
            size_t i_offset = i * 8;
            size_t j_offset = j * 8;
            for (int ii = 0; ii < 8; ii++)
            {
                m_t[(j_offset + 0) * N + i_offset + ii] = matrix[(i_offset + ii) * N + j_offset + 0];
                m_t[(j_offset + 1) * N + i_offset + ii] = matrix[(i_offset + ii) * N + j_offset + 1];
                m_t[(j_offset + 2) * N + i_offset + ii] = matrix[(i_offset + ii) * N + j_offset + 2];
                m_t[(j_offset + 3) * N + i_offset + ii] = matrix[(i_offset + ii) * N + j_offset + 3];
                m_t[(j_offset + 4) * N + i_offset + ii] = matrix[(i_offset + ii) * N + j_offset + 4];
                m_t[(j_offset + 5) * N + i_offset + ii] = matrix[(i_offset + ii) * N + j_offset + 5];
                m_t[(j_offset + 6) * N + i_offset + ii] = matrix[(i_offset + ii) * N + j_offset + 6];
                m_t[(j_offset + 7) * N + i_offset + ii] = matrix[(i_offset + ii) * N + j_offset + 7];
            }
        }
    }
}

template <typename T>
bool verify(const T *m, T *m_t, int M, int N)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (m[i * N + j] != m_t[j * M + i])
            {
                cout << "Error at " << i << " " << j << endl;
                return false;
            }
        }
    }
    clear(m_t, N, N);
    return true;
}

#define TRANSPOSE_ARGS matrix, m_t, N, N

#define MACRO_TEMPLATE(state, T)                                                            \
    {                                                                                       \
        constexpr size_t N = 1024 * 8;                                                      \
        T *matrix = (T *)malloc(N * N * sizeof(T));                                         \
        T *m_t = (T *)malloc(N * N * sizeof(T));                                            \
        for (int i = 0; i < N; i++)                                                         \
        {                                                                                   \
            for (int j = 0; j < N; j++)                                                     \
            {                                                                               \
                matrix[i * N + j] = i * N + j;                                              \
            }                                                                               \
        }                                                                                   \
        QTTBENCH_RUN(state, transposeNative, verify, TRANSPOSE_ARGS)                        \
        constexpr size_t chunk_size = CACHELINE_SIZE / sizeof(T);                           \
        QTTBENCH_RUN(state, (chunkedTranspose<T, 1>), verify, TRANSPOSE_ARGS)               \
        QTTBENCH_RUN(state, (chunkedTranspose<T, 2>), verify, TRANSPOSE_ARGS)               \
        QTTBENCH_RUN(state, (chunkedTranspose<T, 4>), verify, TRANSPOSE_ARGS)               \
        QTTBENCH_RUN(state, (chunkedTranspose<T, 8>), verify, TRANSPOSE_ARGS)               \
        QTTBENCH_RUN(state, (chunkedTranspose<T, chunk_size>), verify, TRANSPOSE_ARGS)      \
        QTTBENCH_RUN(state, (chunkedTranspose<T, chunk_size * 2>), verify, TRANSPOSE_ARGS)  \
        QTTBENCH_RUN(state, (chunkedTranspose<T, chunk_size * 4>), verify, TRANSPOSE_ARGS)  \
        QTTBENCH_RUN(state, (chunkedTranspose<T, chunk_size * 8>), verify, TRANSPOSE_ARGS)  \
        QTTBENCH_RUN(state, (chunkedTranspose<T, chunk_size * 16>), verify, TRANSPOSE_ARGS) \
        QTTBENCH_RUN(state, (chunkedTranspose<T, chunk_size * 32>), verify, TRANSPOSE_ARGS) \
        QTTBENCH_RUN(state, (chunkedTranspose<T, chunk_size * 64>), verify, TRANSPOSE_ARGS) \
        QTTBENCH_RUN(state, chunkedTranspose_unloop_8, verify, TRANSPOSE_ARGS)              \
        state.dump_csv();                                                                   \
        free(matrix);                                                                       \
        free(m_t);                                                                          \
    }

template <typename T>
void test_with_dtype(qttbench::State &state)
{

    constexpr size_t N = 1024 * 8;
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

    // Native transpose
    QTTBENCH_RUN(state, transposeNative, verify, TRANSPOSE_ARGS)

    // chunked transpose
    constexpr size_t chunk_size = CACHELINE_SIZE / sizeof(T);
    // You need to enclose the macro argument in parentheses
    // to avoid macros extract comma in template
    QTTBENCH_RUN(state, (chunkedTranspose<T, 1>), verify, TRANSPOSE_ARGS)
    QTTBENCH_RUN(state, (chunkedTranspose<T, 2>), verify, TRANSPOSE_ARGS)
    QTTBENCH_RUN(state, (chunkedTranspose<T, 4>), verify, TRANSPOSE_ARGS)
    QTTBENCH_RUN(state, (chunkedTranspose<T, 8>), verify, TRANSPOSE_ARGS)
    QTTBENCH_RUN(state, (chunkedTranspose<T, chunk_size>), verify, TRANSPOSE_ARGS)
    // or
    state.run("chunkedTranspose with dtype T, chunk_size * 2 ", [&]()
              { chunkedTranspose<T, chunk_size * 2>(TRANSPOSE_ARGS); }, [&]()
              { return verify(TRANSPOSE_ARGS); });

    QTTBENCH_RUN(state, (chunkedTranspose<T, chunk_size * 4>), verify, TRANSPOSE_ARGS)
    QTTBENCH_RUN(state, (chunkedTranspose<T, chunk_size * 8>), verify, TRANSPOSE_ARGS)
    QTTBENCH_RUN(state, (chunkedTranspose<T, chunk_size * 16>), verify, TRANSPOSE_ARGS)
    QTTBENCH_RUN(state, (chunkedTranspose<T, chunk_size * 32>), verify, TRANSPOSE_ARGS)
    QTTBENCH_RUN(state, (chunkedTranspose<T, chunk_size * 64>), verify, TRANSPOSE_ARGS)

    // unloop transpose
    QTTBENCH_RUN(state, chunkedTranspose_unloop_8, verify, TRANSPOSE_ARGS)

    state.dump_csv();

    free(matrix);
    free(m_t);
}

int main(int argc, char *argv[])
{
    int turns = 1;
    char *turns_t = strutils::getCmdOption(argv, argv + argc, "-t");
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

    qttbench::State state(turns);
    state.set_csv_output(strutils::get_filename_without_extension(__FILE__));

    // Style-1 record with data type
    MACRO_TEMPLATE(state, qttbench::float32_t)
    MACRO_TEMPLATE(state, qttbench::float64_t)

    // Style-2 record without data type
    test_with_dtype<qttbench::int16_t>(state);

    return 0;
}
