#pragma once

#include "ofMain.h"

#include "BaseModel.hpp"
#include "../utils/TorchUtils.hpp"

namespace ofxLibTorch
{

    class Pix2Pix final : public BaseModel
    {
    public:
        Pix2Pix();
        void init(const std::string& modelPath, const int inputH, const int inputW);
        void forward(ofFloatImage& img);
        
        void addNoise(ofFloatImage& img, const float strength);
        void invert(ofFloatImage& img);
    
        static void normalize(at::Tensor& tensor);
        static void denormalize(at::Tensor& tensor);
    
    private:
        int mInputH, mInputW;
    
    };

} // namespace ofxLibTorch
