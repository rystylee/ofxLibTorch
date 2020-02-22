#include "AdaIN.hpp"

namespace ofx {
namespace libtorch {

AdaIN::AdaIN()
{
    float std_data[] = { 0.229, 0.224, 0.225 };
    mStdTensor =  torch::from_blob(std_data, { 3, 1, 1 }, at::TensorOptions(at::kFloat)).to(*mDevice.get());

    float mean_data[] = { 0.485, 0.456, 0.406 };
    mMeanTensor = torch::from_blob(std_data, { 3, 1, 1 }, at::TensorOptions(at::kFloat)).to(*mDevice.get());
}

void AdaIN::init(const std::string& modelPath)
{
    loadModel(modelPath);
}

void AdaIN::forward(const ofFbo& contentFbo, const ofFbo& styleFbo, const float alpha)
{
    torch::NoGradGuard noGrad;

    // Content image
    at::Tensor contentTensor = util::fboToFloatTensor(contentFbo, *mDevice.get());

    // Style image
    at::Tensor styleTensor = util::fboToFloatTensor(styleFbo, *mDevice.get());

    // Normalize
    normalize_(contentTensor);
    normalize_(styleTensor);

    // Alpha factor
    float alpha_data[] = { alpha };
    at::Tensor alpha_tensor = torch::from_blob(alpha_data, at::IntList({ 1 }), at::TensorOptions(at::kFloat)).to(*mDevice.get());

    //
    // Forward process
    // output_tensor { batch, channel, height, width }
    //
    at::Tensor output_tensor = mModule.forward({ contentTensor, styleTensor, alpha_tensor }).toTensor();

    denormalize_(output_tensor);

    // output_tensor { height, width, channel }
    output_tensor = output_tensor.squeeze().permute({ 1, 2, 0 }).to(torch::kCPU);

    float* data = output_tensor.data<float>();

    mImg.setFromPixels(data, output_tensor.size(1), output_tensor.size(0), OF_IMAGE_COLOR, false);
}

void AdaIN::render(const glm::vec2& pos, const float w, const float h)
{
    mImg.draw(pos, w, h);
}

} // namespace libTorch
} // namespace ofx