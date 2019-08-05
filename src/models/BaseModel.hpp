#pragma once

#include <torch/torch.h>
#include <torch/script.h>

namespace ofxLibTorch
{

    class BaseModel
    {
    public:
        BaseModel();
        void loadModel(const std::string& modelPath);
    
    protected:
        std::unique_ptr<torch::Device> mDevice;
        std::shared_ptr<torch::jit::script::Module> mModule;
    
    };

} // namespace ofxLibTorch

