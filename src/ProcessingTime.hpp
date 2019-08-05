#pragma once

#include <iostream>
#include <chrono>

class ProcessingTime
{
public:
    ProcessingTime(const std::string& name)
    {
        mName = name;
        mStart = std::chrono::system_clock::now();
    }

    ~ProcessingTime()
    {
        const auto end = std::chrono::system_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - mStart).count();
        std::cout << mName << elapsed * 0.001 << std::endl;
    }

private:
    std::string mName;
    std::chrono::system_clock::time_point mStart;

};
