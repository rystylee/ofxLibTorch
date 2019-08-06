#pragma once

#include <iostream>
#include <vector>
#include <chrono>

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

        // Time processing
        std::chrono::system_clock::time_point start();
        void end(const std::chrono::system_clock::time_point& start);

    } // namespace util
} // namespace ofxLibTorch
