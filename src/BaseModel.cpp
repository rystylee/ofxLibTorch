#include "BaseModel.hpp"

namespace ofx {
namespace libtorch {

BaseModel::BaseModel()
{
    if (torch::cuda::is_available())
        mDevice = std::make_unique<torch::Device>(torch::kCUDA);
    else
        mDevice = std::make_unique<torch::Device>(torch::kCPU);

    std::cout << "Device: " << *mDevice << std::endl;
}

void BaseModel::loadModel(const std::string& modelPath)
{
    try
    {
        mModule = torch::jit::load(modelPath, *mDevice.get());
        std::cout << "Succeed to load model!" << std::endl;
    }
    catch (const c10::Error& e)
    {
        std::cerr << "[!] Error loading the model" << std::endl;
    }
    mModule.eval();
}

} // namespace libtorch
} // namespace ofx