#include "Utils.hpp"

namespace ofx {
namespace libtorch {
namespace util {

std::chrono::system_clock::time_point now()
{
    return std::chrono::system_clock::now();
}

void end(const std::chrono::system_clock::time_point& start)
{
    const std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "elapsed: " << elapsed * 0.001 << std::endl;
}

} // namespace util
} // namespace libTorch
} // namespace ofx