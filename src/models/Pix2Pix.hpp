#pragma once

#include "ofMain.h"

#include "BaseModel.hpp"
#include "../utils/TorchUtils.hpp"

namespace ofx {
namespace libtorch {

class Pix2Pix final : public BaseModel
{
public:
    Pix2Pix();
    void init(const std::string& modelPath, const int inputH, const int inputW);
    void forward(ofFloatImage& img);
    
    void addNoise(ofFloatImage& img, const float strength);
    void invert(ofFloatImage& img);

    static void normalize_(at::Tensor& tensor)
    {
        tensor.mul_(2.0).sub_(1.0);
    }

    static void denormalize_(at::Tensor& tensor)
    {
        tensor.add_(1.0).mul_(0.5);
    }

private:
    int mInputH, mInputW;

};

} // namespace libTorch
} // namespace ofx
