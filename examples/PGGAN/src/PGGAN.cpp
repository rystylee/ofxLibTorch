#include "PGGAN.hpp"

//--------------------------------------------------------------
// Public
//--------------------------------------------------------------
PGGAN::PGGAN()
{
}

PGGAN::PGGAN(const std::string& modelPath)
{
    init(modelPath);
}

void PGGAN::init(const std::string& modelPath)
{
    loadModel(modelPath);

    mNoiseTensor = torch::randn({ mBatchSize, mDimZ }, at::TensorOptions(at::kFloat)).to(*mDevice.get());
    mStartNoiseTensor = torch::randn({ mBatchSize, mDimZ }, at::TensorOptions(at::kFloat)).to(*mDevice.get());
    mGoalNoiseTensor = torch::randn({ mBatchSize, mDimZ }, at::TensorOptions(at::kFloat)).to(*mDevice.get());
}

void PGGAN::forward(ofFloatImage& img)
{
    torch::NoGradGuard noGrad;

    //
    // Forward process
    // output_tensor { batch, channel, height, width }
    //
    at::Tensor outputTensor = mModule.forward({ mNoiseTensor }).toTensor();

    // Denormalize
    PGGAN::denormalize_(outputTensor);

    // outputTensor { height, width, channel }
    outputTensor = outputTensor.squeeze().permute({ 1, 2, 0 }).to(torch::kCPU);

    float* data = outputTensor.data<float>();

    img.setFromPixels(data, outputTensor.size(1), outputTensor.size(0), OF_IMAGE_COLOR, false);
}

void PGGAN::stepNoise()
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

//--------------------------------------------------------------
// Private
//--------------------------------------------------------------

