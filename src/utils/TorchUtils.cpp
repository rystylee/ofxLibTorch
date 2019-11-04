#include "TorchUtils.hpp"

namespace ofx {
namespace libtorch {
namespace util {

at::Tensor fboToFloatTensor(const ofFbo& fbo, const torch::Device& device)
{
    ofFloatPixels pix;
    fbo.readToPixels(pix);
    cv::Mat mat = toCv(pix);
    at::Tensor tensor = torch::from_blob(mat.data, at::IntList({ mat.rows, mat.cols, mat.channels() }), at::TensorOptions(at::kFloat)).to(device);
    tensor = tensor.permute({ 2, 0, 1 }).unsqueeze(0);
    return tensor;
}

at::Tensor floatPixelsToFloatTensor(const ofFloatPixels& pix, const torch::Device& device)
{
    ofFloatPixels p = pix;
    cv::Mat mat = toCv(p);
    at::Tensor tensor = torch::from_blob(mat.data, at::IntList({ mat.rows, mat.cols, mat.channels() }), at::TensorOptions(at::kFloat)).to(device);
    tensor = tensor.permute({ 2, 0, 1 }).unsqueeze(0);
    return tensor;
}

} // namespace util
} // namespace libtorch
} // namespace ofx