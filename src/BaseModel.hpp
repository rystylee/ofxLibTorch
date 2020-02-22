#pragma once

#include <torch/torch.h>
#include <torch/script.h>

namespace ofx {
namespace libtorch {

class BaseModel
{
public:
    BaseModel();
    void loadModel(const std::string& modelPath);

protected:
    std::unique_ptr<torch::Device> mDevice;
    torch::jit::script::Module mModule;

};

} // namespace libTorch
} // namespace ofx
