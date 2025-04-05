#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

torch::Tensor create_fp8_from_uint8(torch::Tensor uint8_data) {
    TORCH_CHECK(uint8_data.device().is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(uint8_data.dtype() == torch::kUInt8, "Input tensor must be uint8");
    
    uint8_t* data_ptr = uint8_data.data_ptr<uint8_t>();
    
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat8_e4m3fn)
        .device(torch::kCUDA);
    
    auto size = uint8_data.sizes().vec();
    torch::Tensor fp8_tensor = torch::from_blob(
        data_ptr,
        size,
        /*deleter=*/[](void*) {},
        options
    );

    return fp8_tensor.clone();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("create_fp8_from_uint8", &create_fp8_from_uint8, 
          "Create an FP8 tensor from uint8 data");
}