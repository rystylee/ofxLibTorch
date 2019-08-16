#pragma once

#include "ofJson.h"
#include "ofFbo.h"

#include "BaseModel.hpp"
#include "../utils/TorchUtils.hpp"

namespace ofxLibTorch
{

    class VGG16 final : public BaseModel
    {
    public:
        VGG16();
        void init(const std::string& modelPath);
        void forward(const ofFbo& inputFbo);

        inline void preprocess_(torch::Tensor& tensor)
        {
            tensor.sub_(mMeanTensor).div_(mStdTensor);
        }

    private:
        void loadJson(const std::string& filePath);

        std::map<int, std::string> mClassLabels;

        at::Tensor mStdTensor;
        at::Tensor mMeanTensor;

    };

} // namespace ofxLibTorch
