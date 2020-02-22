#include "BicycleGAN.hpp"

namespace ofx {
namespace libtorch {

BicycleGAN::BicycleGAN()
{
}

BicycleGAN::BicycleGAN(const std::string& modelPath)
{
    init(modelPath);
}

void BicycleGAN::init(const std::string& modelPath)
{
    loadModel(modelPath);

    mNoiseTensor = torch::randn({ mBatchSize, mDimZ }, at::TensorOptions(at::kFloat)).to(*mDevice.get());
    mStartNoiseTensor = torch::randn({ mBatchSize, mDimZ }, at::TensorOptions(at::kFloat)).to(*mDevice.get());
    mGoalNoiseTensor = torch::randn({ mBatchSize, mDimZ }, at::TensorOptions(at::kFloat)).to(*mDevice.get());
}

void BicycleGAN::forward(ofFloatImage& img)
{
    torch::NoGradGuard no_grad;

    cv::Mat inputMat = util::toCv(img.getPixels());
    at::Tensor inputTensor = torch::from_blob(inputMat.data, at::IntList({ inputMat.rows, inputMat.cols, inputMat.channels() }), at::TensorOptions(at::kFloat)).to(*mDevice.get());
    inputTensor = inputTensor.permute({ 2, 0, 1 }).unsqueeze(0);

    // Normalize
    BicycleGAN::normalize_(inputTensor);

    //
    // Forward process
    // output_tensor { batch, channel, height, width }
    //
    at::Tensor output_tensor = mModule.forward({ inputTensor, mNoiseTensor }).toTensor();

    // Denormalize
    BicycleGAN::denormalize_(output_tensor);

    // outputTensor { height, width, channel }
    output_tensor = output_tensor.squeeze().permute({ 1, 2, 0 }).to(torch::kCPU);

    float* data = output_tensor.data<float>();

    img.setFromPixels(data, output_tensor.size(1), output_tensor.size(0), OF_IMAGE_COLOR, false);
}

void BicycleGAN::stepNoise()
{
    if (mNoiseStepCounter > mNoiseMaxSteps)
    {
        mNoiseStepCounter = 0.0f;
        mStartNoiseTensor = mGoalNoiseTensor;
        mGoalNoiseTensor = torch::randn({ mBatchSize, mDimZ }, at::TensorOptions(at::kFloat)).to(*mDevice.get());
    }

    at::Tensor diff = mGoalNoiseTensor.sub(mStartNoiseTensor);
    diff.mul_(mNoiseStepCounter / mNoiseMaxSteps);
    mNoiseTensor = mStartNoiseTensor.add(diff);

    mNoiseStepCounter += 1.0f;
}

void BicycleGAN::addNoise(ofFloatImage& img, const float strength = 0.1)
{
    auto data = img.getPixels();
    for (auto& d : data)
    {
        float n = 0.0;
        n = ofNoise(d * 0.1, ofGetElapsedTimef() * 0.01);
        d += n * strength;
        d = ofClamp(d, 0.0, 1.0);
    }
    img.setFromPixels(data.getPixels(), img.getWidth(), img.getHeight(), OF_IMAGE_COLOR, true);
}

void BicycleGAN::invert(ofFloatImage& img)
{
    auto data = img.getPixels();
    for (auto& d : data)
    {
        d = 1.0 - d;
        d = ofClamp(d, 0.0, 1.0);
    }
    img.setFromPixels(data.getPixels(), img.getWidth(), img.getHeight(), OF_IMAGE_COLOR, true);
}

} // namespace libTorch
} // namespace ofx