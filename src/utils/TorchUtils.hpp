#pragma once

#include "ofFbo.h"
#include "ofImage.h"

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

namespace ofx {
namespace libtorch {
namespace util {

at::Tensor fboToFloatTensor(const ofFbo& fbo, const torch::Device& device);
at::Tensor floatPixelsToFloatTensor(const ofFloatPixels& pix, const torch::Device& device);

// OpenCV utils
template <typename T>
cv::Mat toCv(ofPixels_<T>& pix)
{
    int depth;
    switch(pix.getBytesPerChannel())
    {
        case 4: depth = CV_32F;
        case 2: depth = CV_16U;
        case 1: default: depth = CV_8U;
    }
    return cv::Mat(pix.getHeight(), pix.getWidth(), CV_MAKETYPE(depth, pix.getNumChannels()), pix.getData(), 0);
}

} // namespace util
} // namespace libtorch
} // namespace ofx
