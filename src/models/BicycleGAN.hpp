#pragma once

#include "ofMain.h"

#include "BaseModel.hpp"
#include "../utils/TorchUtils.hpp"

namespace ofx {
namespace libtorch {

class BicycleGAN final : public BaseModel
{
public:
    BicycleGAN();
    BicycleGAN(const std::string& modelPath);
    void init(const std::string& modelPath);
    void forward(ofFloatImage& img);

    void stepNoise();

    static void addNoise(ofFloatImage& img, const float strength);
    static void invert(ofFloatImage& img);

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

} // namespace libTorch
} // namespace ofx
