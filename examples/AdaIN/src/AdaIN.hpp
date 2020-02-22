#pragma once

#include "ofMain.h"

#include "ofxLibTorch.h"

class AdaIN final : ofxLibTorch::BaseModel
{
public:
    AdaIN();
    void init(const std::string& modelPath);
    void forward(const ofFbo& contentFbo, const ofFbo& styleFbo, const float alpha);
    void render(const glm::vec2& pos, const float w, const float h);

    void normalize_(at::Tensor& tensor)
    {
        tensor.sub_(mMeanTensor).div_(mStdTensor);
    }

    void denormalize_(at::Tensor& tensor)
    {
        torch::clamp_(tensor.mul_(mStdTensor).add_(mMeanTensor), 0, 1);
    }

private:
    at::Tensor mStdTensor;
    at::Tensor mMeanTensor;

    ofFloatImage mImg;

    //FastFboReader mFastFboReader;
};
