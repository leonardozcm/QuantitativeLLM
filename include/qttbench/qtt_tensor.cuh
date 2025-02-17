#ifndef QTT_TENSOR_H
#define QTT_TENSOR_H

#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>
#include <vector>
#include <algorithm>

#include "qtt_cuda_runtime.cuh"
#include <curand_kernel.h>

namespace qttbench {

#define MAX_DIM 4

__global__ void initRandomArray(float* arr, int width, int height, unsigned long seed) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = y * width + x;

    if (x < width && y < height) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        arr[idx] = curand_uniform(&state) * 20.0f - 10.0f; // Scale to [-10.0, 10.0]
    }
};

template<typename T>
class Tensor {
public:
    size_t ne[MAX_DIM];
    size_t nb[MAX_DIM];

    Tensor(size_t dim, std::vector<int> ne_, bool is_gpu = true);
    Tensor(size_t dim, std::vector<int> ne_, const T* array, bool is_gpu = true);
    ~Tensor();

    void initialize_random();
    T* data_ptr();
    T element_wise_difference(const Tensor& tensor) const;
    Tensor operator+(const Tensor& tensor) const;
    Tensor operator-(const Tensor& tensor) const;
    Tensor operator*(const Tensor& tensor) const;
    Tensor operator/(const Tensor& tensor) const;
    void cpu_data(T* data_ptr) const;
    void save_to_file(const std::string& filename) const;
    static Tensor* load_from_file(const std::string& filename);

    template<typename... Args>
    void print(Args... args) const;

    template<typename... Args>
    T& operator()(Args... args);

    template<typename... Args>
    const T& operator()(Args... args) const;

private:
    size_t dim_;
    size_t size_;
    bool is_gpu_;
    T* data_;

    void allocate_memory();
    void free_memory();
    void initialize_random_cpu();
    void initialize_random_gpu();
    void initialize_from_array(const T* array);
    T element_wise_difference_cpu(const Tensor& tensor) const;
    void element_wise_add_cpu(const Tensor& tensor, Tensor& result) const;
    void element_wise_sub_cpu(const Tensor& tensor, Tensor& result) const;
    void element_wise_mul_cpu(const Tensor& tensor, Tensor& result) const;
    void element_wise_div_cpu(const Tensor& tensor, Tensor& result) const;

    void print_recursive(size_t dim, size_t offset) const;

    template<typename... Args>
    size_t compute_offset(Args... args) const;
};

template<typename T>
Tensor<T>::Tensor(size_t dim, std::vector<int> ne_, bool is_gpu)
    : dim_(dim), is_gpu_(is_gpu), data_(nullptr) {
    assert(dim <= MAX_DIM);
    size_ = 1;
    for (size_t i = 0; i < dim; ++i) {
        ne[i] = ne_[i];
        nb[i] = (i == 0) ? sizeof(T) : nb[i-1] * ne[i-1];
        size_ *= ne[i];
    }
    allocate_memory();
}

template<typename T>
Tensor<T>::Tensor(size_t dim, std::vector<int> ne_, const T* array, bool is_gpu)
    : dim_(dim), is_gpu_(is_gpu), data_(nullptr) {
    assert(dim <= MAX_DIM);
    size_ = 1;
    for (size_t i = 0; i < dim; ++i) {
        ne[i] = ne_[i];
        nb[i] = (i == 0) ? sizeof(T) : nb[i-1] * ne[i-1];
        size_ *= ne[i];
    }
    allocate_memory();
    initialize_from_array(array);
}

template<typename T>
Tensor<T>::~Tensor() {
    free_memory();
}

template<typename T>
T* Tensor<T>::data_ptr() {
    return data_;
}


template<typename T>
void Tensor<T>::cpu_data(T* data_ptr) const {
    cudaMemcpy(data_ptr, data_, size_ * sizeof(T), cudaMemcpyDeviceToHost);
}

template<typename T>
void Tensor<T>::allocate_memory() {
    if (!is_gpu_) {
        data_ = new T[size_];
    } else {
        cudaMalloc(&data_, size_ * sizeof(T));
    }
}

template<typename T>
void Tensor<T>::free_memory() {
    if (!is_gpu_) {
        delete[] data_;
    } else {
        cudaFree(data_);
    }
}

template<typename T>
void Tensor<T>::initialize_random() {
    if (!is_gpu_) {
        initialize_random_cpu();
    } else {
        initialize_random_gpu();
    }
}

template<typename T>
void Tensor<T>::initialize_random_cpu() {
#pragma omp parallel for
    for (size_t i = 0; i < size_; ++i) {
        data_[i] = static_cast<T>(rand()) / RAND_MAX;
    }
}

template<typename T>
void Tensor<T>::initialize_random_gpu() {


    int width = ne[0];
    int height = size_ / width;
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    initRandomArray<<<blocksPerGrid, threadsPerBlock>>>(data_, width, height, time(NULL));
}


template<typename T>
void Tensor<T>::initialize_from_array(const T* array) {
    if (!is_gpu_) {
        std::copy(array, array + size_, data_);
    } else {
        cudaMemcpy(data_, array, size_ * sizeof(T), cudaMemcpyHostToDevice);
    }
}

template<typename T>
Tensor<T> Tensor<T>::operator+(const Tensor& tensor) const {
    assert(size_ == tensor.size_);
    Tensor result(dim_, ne, is_gpu_);
    if (!is_gpu_) {
        element_wise_add_cpu(tensor, result);
    } else {
        std::cerr << "GPU support needs to be implemented." << std::endl;
    }
    return result;
}

template<typename T>
Tensor<T> Tensor<T>::operator-(const Tensor& tensor) const {
    assert(size_ == tensor.size_);
    Tensor result(dim_, ne, is_gpu_);
    if (!is_gpu_) {
        element_wise_sub_cpu(tensor, result);
    } else {
        std::cerr << "GPU support needs to be implemented." << std::endl;
    }
    return result;
}

template<typename T>
Tensor<T> Tensor<T>::operator*(const Tensor& tensor) const {
    assert(size_ == tensor.size_);
    Tensor result(dim_, ne, is_gpu_);
    if (!is_gpu_) {
        element_wise_mul_cpu(tensor, result);
    } else {
        std::cerr << "GPU support needs to be implemented." << std::endl;
    }
    return result;
}

template<typename T>
Tensor<T> Tensor<T>::operator/(const Tensor& tensor) const {
    assert(size_ == tensor.size_);
    Tensor result(dim_, ne, is_gpu_);
    if (!is_gpu_) {
        element_wise_div_cpu(tensor, result);
    } else {
        std::cerr << "GPU support needs to be implemented." << std::endl;
    }
    return result;
}

template<typename T>
T Tensor<T>::element_wise_difference(const Tensor& tensor) const {
    assert(size_ == tensor.size_);
    T diff = 0;
    if (!is_gpu_) {
        diff = element_wise_difference_cpu(tensor);
    } else {
        std::cerr << "GPU support needs to be implemented." << std::endl;
    }
    return diff;
}

template<typename T>
T Tensor<T>::element_wise_difference_cpu(const Tensor& tensor) const {
    T diff = 0;
    for (size_t i = 0; i < size_; ++i) {
        diff += std::abs(data_[i] - tensor.data_[i]);
    }
    return diff;
}

template<typename T>
void Tensor<T>::element_wise_add_cpu(const Tensor& tensor, Tensor& result) const {
    for (size_t i = 0; i < size_; ++i) {
        result.data_[i] = data_[i] + tensor.data_[i];
    }
}

template<typename T>
void Tensor<T>::element_wise_sub_cpu(const Tensor& tensor, Tensor& result) const {
    for (size_t i = 0; i < size_; ++i) {
        result.data_[i] = data_[i] - tensor.data_[i];
    }
}

template<typename T>
void Tensor<T>::element_wise_mul_cpu(const Tensor& tensor, Tensor& result) const {
    for (size_t i = 0; i < size_; ++i) {
        result.data_[i] = data_[i] * tensor.data_[i];
    }
}

template<typename T>
void Tensor<T>::element_wise_div_cpu(const Tensor& tensor, Tensor& result) const {
    for (size_t i = 0; i < size_; ++i) {
        result.data_[i] = data_[i] / tensor.data_[i];
    }
}

template<typename T>
void Tensor<T>::save_to_file(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }

    file.write(reinterpret_cast<const char*>(&dim_), sizeof(dim_));
    file.write(reinterpret_cast<const char*>(ne), sizeof(ne));
    file.write(reinterpret_cast<const char*>(nb), sizeof(nb));

    if (!is_gpu_) {
        file.write(reinterpret_cast<const char*>(data_), size_ * sizeof(T));
    } else {
        std::cerr << "GPU support needs to be implemented." << std::endl;
    }

    file.close();
}

template<typename T>
Tensor<T>* Tensor<T>::load_from_file(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file for reading: " << filename << std::endl;
        return nullptr;
    }

    size_t dim;
    size_t ne[MAX_DIM];
    size_t nb[MAX_DIM];
    file.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    file.read(reinterpret_cast<char*>(ne), sizeof(ne));
    file.read(reinterpret_cast<char*>(nb), sizeof(nb));

    Tensor* tensor = new Tensor(dim, ne, false);
    tensor->allocate_memory();

    if (!tensor->is_gpu_) {
        file.read(reinterpret_cast<char*>(tensor->data_), tensor->size_ * sizeof(T));
    } else {
        std::cerr << "GPU support needs to be implemented." << std::endl;
        delete tensor;
        return nullptr;
    }

    file.close();
    return tensor;
}

template<typename T>
template<typename... Args>
T& Tensor<T>::operator()(Args... args) {
    size_t offset = compute_offset(args...);
    return data_[offset];
}

template<typename T>
template<typename... Args>
const T& Tensor<T>::operator()(Args... args) const {
    size_t offset = compute_offset(args...);
    return data_[offset];
}

template<typename T>
template<typename... Args>
size_t Tensor<T>::compute_offset(Args... args) const {
    assert(sizeof...(args) <= dim_);
    size_t indices[] = {static_cast<size_t>(args)...};
    size_t offset = 0;
    for (size_t i = 0; i < sizeof...(args); ++i) {
        if (indices[i] >= ne[dim_ - 1 - i]) {
            std::cerr << "Index " << indices[i] << " exceeds the number of elements in dimension " << i << std::endl;
            std::exit(EXIT_FAILURE);
        }
        offset += indices[i] * nb[dim_ - 1 - i] / sizeof(T);
    }
    return offset;
}

template<typename T>
template<typename... Args>
void Tensor<T>::print(Args... args) const {
    if (sizeof...(args) > dim_) {
        std::cerr << "Number of indices exceeds the tensor dimensions." << std::endl;
        return;
    }

    if (!is_gpu_) {
        if (sizeof...(args) == 0) {
            print_recursive(0, 0);
        } else {
            size_t offset = compute_offset(args...);
            print_recursive(sizeof...(args), offset);
        }
        std::cout << std::endl;
    } else {
        std::cerr << "GPU support needs to be implemented." << std::endl;
    }
}

template<typename T>
void Tensor<T>::print_recursive(size_t dim, size_t offset) const {
    if (dim == dim_) {
        std::cout << "[" << data_[offset] << "]";
    } else if (dim == dim_ - 1) {
        std::cout << "[";
        for (size_t i = 0; i < ne[dim_ - 1 - dim]; ++i) {
            std::cout << data_[offset + i];
            if (i < ne[dim_ - 1 - dim] - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]";
    } else {
        std::cout << "[";
        for (size_t i = 0; i < ne[dim_ - 1 - dim]; ++i) {
            if (i > 0) {
                std::cout << ",";
            }
            print_recursive(dim + 1, offset + i * nb[dim_ - 1 - dim] / sizeof(T));
        }
        std::cout << "]";
    }
}

} // namespace qttbench

#endif // QTT_TENSOR_H
