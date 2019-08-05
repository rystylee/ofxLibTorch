#pragma once

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

namespace ofxLibTorch
{
    namespace util
    {

        template<typename T>
        void map(std::vector<std::vector<T>>& values, const T inputMin, const T inputMax, const T outputMin, const T outputMax)
        {
            for (auto& batch : values)
            {
                map(batch, inputMin, inputMax, outputMin, outputMax);
            }
        }
        
        template<typename T>
        void map(std::vector<T>& values, const T inputMin, const T inputMax, const T outputMin, const T outputMax)
        {
            for (auto& v : values)
            {
                v = map(v, inputMin, inputMax, outputMin, outputMax);
            }
        }
        
        template<typename T>
        void map(T* values, int64_t size, const T inputMin, const T inputMax, const T outputMin, const T outputMax)
        {
            for (int64_t i = 0; i < size; i++)
            {
                values[i] = map(values[i], inputMin, inputMax, outputMin, outputMax);
            }
        }
        
        template<typename T>
        T map(const T value, const T inputMin, const T inputMax, const T outputMin, const T outputMax)
        {
            return ((value - inputMin) / (inputMax - inputMin) * (outputMax - outputMin) + outputMin);
        }

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
} // namespace ofxLibTorch
