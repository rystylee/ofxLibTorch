#pragma once

#include "ofMain.h"

#include "BaseModel.hpp"

namespace ofxLibTorch
{

    class BigGAN final : public BaseModel
    {
    public:
        BigGAN();
        BigGAN(const std::string& modelPath);
        void init(const std::string& modelPath);
        void forward(ofFloatImage& img, const float truncation=0.4f);
        //void forward(ofFloatImage& img, const int classIndex, const float truncation=0.4f);
        void stepClass();
        void stepNoise(const float truncation=0.4f);

        static inline void denormalize_(at::Tensor& tensor)
        {
            torch::clamp_(tensor.add_(1.0).mul_(0.5), 0, 1);
        }

    private:
        at::Tensor truncatedNoiseSample(c10::IntArrayRef size, const float truncation);
        at::Tensor oneHotFromInt(const int index);

        const int mBatchSize { 1 };
        static const int mNumClasses { 1000 };
        const int mDimZ { 128 };

        // Latent vector
        at::Tensor mNoiseTensor;
        at::Tensor mStartNoiseTensor;
        at::Tensor mGoalNoiseTensor;

        float mNoiseStepCounter { 0.0f };
        const float mNoiseMaxSteps { 60.0f };

        // Class vector
        at::Tensor mClassTensor;
        at::Tensor mStartClassTensor;
        at::Tensor mGoalClassTensor;

        float mClassStepCounter { 0.0f };
        const float mClassMaxSteps { 60.0f };
    };

}
