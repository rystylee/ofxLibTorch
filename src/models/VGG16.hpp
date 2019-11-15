#pragma once

#include "ofJson.h"
#include "ofFbo.h"

#include "BaseModel.hpp"
#include "../utils/TorchUtils.hpp"

namespace ofx {
namespace libtorch {

class VGG16 final : public BaseModel
{
public:
    VGG16();
    void init(const std::string& modelPath);

    template <typename T>
    T forward(const ofFbo& inputFbo);

    template<>
    std::pair<int, float> forward(const ofFbo& inputFbo)
    {
        torch::NoGradGuard noGrad;

        at::Tensor inputTensor = util::fboToFloatTensor(inputFbo, *mDevice.get());

        preprocess_(inputTensor);

        //
        // Forward process
        // outputTensor { batch, 1000 }
        //
        at::Tensor outputTensor = mModule->forward({ inputTensor }).toTensor();

        at::Tensor out = torch::softmax(outputTensor, 1);
        auto maxId = torch::argmax(out).item();
        auto maxProb = torch::max(out).item();

        std::cout << "Probability: " << maxProb << std::endl;
        std::cout << "Label: " << mClassLabels[maxId.toInt()] << std::endl;

        return std::make_pair(maxId.toInt(), maxProb.toFloat());
    }

    template<>
    std::pair<int, at::Tensor> forward(const ofFbo& inputFbo)
    {
        torch::NoGradGuard noGrad;

        at::Tensor inputTensor = util::fboToFloatTensor(inputFbo, *mDevice.get());

        preprocess_(inputTensor);

        //
        // Forward process
        // outputTensor { batch, 1000 }
        //
        at::Tensor outputTensor = mModule->forward({ inputTensor }).toTensor();

        at::Tensor out = torch::softmax(outputTensor, 1);
        auto maxId = torch::argmax(out).item();
        auto maxProb = torch::max(out).item();

        //std::cout << "Probability: " << maxProb << std::endl;
        //std::cout << "Label: " << mClassLabels[maxId.toInt()] << std::endl;

        return std::make_pair(maxId.toInt(), outputTensor.squeeze().to(torch::kCPU));
    }

    template <typename T>
    T forward(const ofFloatPixels& inputPixels);

    template<>
    std::pair<int, at::Tensor> forward(const ofFloatPixels& inputPixel)
    {
        torch::NoGradGuard noGrad;

        at::Tensor inputTensor = util::floatPixelsToFloatTensor(inputPixel, *mDevice.get());

        preprocess_(inputTensor);

        //
        // Forward process
        // outputTensor { batch, 1000 }
        //
        at::Tensor outputTensor = mModule->forward({ inputTensor }).toTensor();

        at::Tensor out = torch::softmax(outputTensor, 1);
        auto maxId = torch::argmax(out).item();
        auto maxProb = torch::max(out).item();

        //std::cout << "Probability: " << maxProb << std::endl;
        //std::cout << "Label: " << mClassLabels[maxId.toInt()] << std::endl;

        return std::make_pair(maxId.toInt(), outputTensor.squeeze().to(torch::kCPU));
    }

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

} // namespace libtorch
} // namespace ofx
