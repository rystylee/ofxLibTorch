#include "BaseModel.hpp"

namespace ofxLibTorch
{

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
        mModule = torch::jit::load(modelPath, *mDevice.get());
        mModule->eval();

        if (mModule == nullptr)
            std::cerr << "Error: Failed to load Torch Script Module..." << std::endl;
        else
            std::cout << "Succeed to load Torch Script Module!" << std::endl;
    }

} // namespace ofxLibTorch