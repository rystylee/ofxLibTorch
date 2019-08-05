#pragma once

#include "ofMain.h"

#include <torch/torch.h>
#include <torch/script.h>

#include "../Utilities.hpp"
#include "../FastFboReader.hpp"
#include "../ProcessingTime.hpp"

namespace ofxLibTorch
{

    class AdaIN
    {
    public:
        AdaIN();
        void init(const std::string& modelPath);
        void forward(const ofFbo& contentFbo, const ofFbo& styleFbo, const float alpha);

        void normalize_(at::Tensor& tensor);
        void denormalize_(at::Tensor& tensor);

    private:
        std::unique_ptr<torch::Device> mDevice;
        std::shared_ptr<torch::jit::script::Module> mModule;

        at::Tensor mStdTensor;
        at::Tensor mMeanTensor;

        FastFboReader mFastFboReader;
    };

} // namespace ofxLibTorch
