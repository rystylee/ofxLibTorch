#pragma once

#include "ofMain.h"

#include <torch/torch.h>
#include <torch/script.h>
#include "../Utilities.hpp"

namespace ofxLibTorch
{

    class Pix2Pix
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
        std::unique_ptr<torch::Device> mDevice;
        std::shared_ptr<torch::jit::script::Module> mModule;
        
        // (h, w)
        int mInputH, mInputW;
    
    };

} // namespace ofxLibTorch
