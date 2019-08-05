#pragma once

#include "ofMain.h"

#include "BaseModel.hpp"
#include "../Utilities.hpp"

namespace ofxLibTorch
{

    class BicycleGAN final : public BaseModel
    {
    public:
        BicycleGAN();
        BicycleGAN(const std::string& modelPath);
        void init(const std::string& modelPath);
        void forward(ofFloatImage& img);

        void stepNoise();

        static inline void normalize_(at::Tensor& tensor)
        {
            tensor.mul_(2.0).sub_(1.0);
        }

        static inline void denormalize_(at::Tensor& tensor)
        {
            tensor.add_(1.0).mul_(0.5);
        }

    private:
        const int mBatchSize { 1 };
        const int mDimZ { 8 };

        // Latent vector
        at::Tensor mNoiseTensor;
        at::Tensor mStartNoiseTensor;
        at::Tensor mGoalNoiseTensor;

        float mNoiseStepCounter { 0.0f };
        const float mNoiseMaxSteps { 60.0f };
    };

} // namespace ofxLibTorch
