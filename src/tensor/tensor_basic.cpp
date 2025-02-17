#include "qtt_tensor.cuh"
#include <vector>

int main() {
    std::vector<int> ne = {5, 4, 3, 2};
    float array[120];
    for (size_t i = 0; i < 120; ++i) {
        array[i] = static_cast<float>(i);
    }

    qttbench::Tensor<float> tensor1(4, ne, false);
    qttbench::Tensor<float> tensor2(4, ne, array, false);

    tensor1.initialize_random();

    float diff = tensor1.element_wise_difference(tensor2);
    std::cout << "Element-wise difference: " << diff << std::endl;

    qttbench::Tensor<float> tensor3 = tensor1 + tensor2;
    qttbench::Tensor<float> tensor4 = tensor1 - tensor2;
    qttbench::Tensor<float> tensor5 = tensor1 * tensor2;
    qttbench::Tensor<float> tensor6 = tensor1 / tensor2;

    std::cout << "Tensor1: " << std::endl;
    tensor1.print();
    std::cout << "Tensor2: " << std::endl;
    tensor2.print();
    std::cout << "Tensor3 (Tensor1 + Tensor2): " << std::endl;
    tensor3.print();
    std::cout << "Tensor4 (Tensor1 - Tensor2): " << std::endl;
    tensor4.print();
    std::cout << "Tensor5 (Tensor1 * Tensor2): " << std::endl;
    tensor5.print();
    std::cout << "Tensor6 (Tensor1 / Tensor2): " << std::endl;
    tensor6.print();

    tensor1.save_to_file("tensor1.dat");

    qttbench::Tensor<float>* tensor_loaded = qttbench::Tensor<float>::load_from_file("tensor1.dat");

    if (tensor_loaded) {
        std::cout << "Tensor loaded from file: " << std::endl;
        tensor_loaded->print();

        tensor_loaded->print(1, 2, 3);

        (*tensor_loaded)(1, 2, 3, 4) = 42.0f;
        std::cout << "Modified tensor: " << std::endl;
        tensor_loaded->print();

        delete tensor_loaded;
    }

    return 0;
}
