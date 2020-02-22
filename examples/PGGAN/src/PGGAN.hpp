#pragma once

#include "ofMain.h"
#include "ofxLibTorch.h"

class PGGAN final : public ofxLibTorch::BaseModel 
{
public:
    PGGAN();
    PGGAN(const std::string& modelPath);
    void init(const std::string& modelPath);
    void forward(ofFloatImage& img);
    void stepNoise();

    static inline void denormalize_(at::Tensor& tensor)
    {
        torch::clamp_(tensor.add_(1.0).mul_(0.5), 0, 1);
    }

private:
    const int mBatchSize { 1 };
    const int mDimZ { 512 };

    // Latent vector
    at::Tensor mNoiseTensor;
    at::Tensor mStartNoiseTensor;
    at::Tensor mGoalNoiseTensor;

    float mNoiseStepCounter { 0.0f };
    const float mNoiseMaxSteps { 60.0f };

};
