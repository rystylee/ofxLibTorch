#include "BigGAN.hpp"

//--------------------------------------------------------------
// Public
//--------------------------------------------------------------
BigGAN::BigGAN()
{
}

BigGAN::BigGAN(const std::string& modelPath)
{
    init(modelPath);
}

void BigGAN::init(const std::string& modelPath)
{
    loadModel(modelPath);

    mStartNoiseTensor = truncatedNoiseSample({ mBatchSize, mDimZ }, 0.4f);
    mNoiseTensor = truncatedNoiseSample({ mBatchSize, mDimZ }, 0.4f);
    mGoalNoiseTensor = truncatedNoiseSample({ mBatchSize, mDimZ }, 0.4f);

    mStartClassTensor = oneHotFromInt(static_cast<int>(ofRandom(mNumClasses)));
    mClassTensor = oneHotFromInt(static_cast<int>(ofRandom(mNumClasses)));
    mGoalClassTensor = oneHotFromInt(static_cast<int>(ofRandom(mNumClasses)));
}

void BigGAN::forward(ofFloatImage& img, const float truncation)
{
    torch::NoGradGuard noGrad;

    float truncation_data[] = { truncation };
    at::Tensor truncationTensor = torch::from_blob(truncation_data, at::IntList({ 1 }), at::TensorOptions(at::kFloat)).to(*mDevice.get());

    //
    // Forward process
    // output_tensor { batch, channel, height, width }
    //
    at::Tensor outputTensor = mModule.forward({ mNoiseTensor, mClassTensor, truncationTensor }).toTensor();

    // Denormalize
    BigGAN::denormalize_(outputTensor);

    // outputTensor { height, width, channel }
    outputTensor = outputTensor.squeeze().permute({ 1, 2, 0 }).to(torch::kCPU);

    float* data = outputTensor.data<float>();

    img.setFromPixels(data, outputTensor.size(1), outputTensor.size(0), OF_IMAGE_COLOR, false);
}

void BigGAN::forward(ofFloatImage& img, const int classIndex, const float truncation)
{
    torch::NoGradGuard noGrad;

    float truncation_data[] = { truncation };
    at::Tensor truncationTensor = torch::from_blob(truncation_data, at::IntList({ 1 }), at::TensorOptions(at::kFloat)).to(*mDevice.get());

    at::Tensor classTensor = oneHotFromInt(classIndex);

    //
    // Forward process
    // output_tensor { batch, channel, height, width }
    //
    at::Tensor outputTensor = mModule.forward({ mNoiseTensor, classTensor, truncationTensor }).toTensor();

    // Denormalize
    BigGAN::denormalize_(outputTensor);

    // outputTensor { height, width, channel }
    outputTensor = outputTensor.squeeze().permute({ 1, 2, 0 }).to(torch::kCPU);

    float* data = outputTensor.data<float>();

    img.setFromPixels(data, outputTensor.size(1), outputTensor.size(0), OF_IMAGE_COLOR, false);
}

void BigGAN::forward(ofFloatImage& img, at::Tensor& classTensor, const float truncation)
{
    torch::NoGradGuard noGrad;

    float truncation_data[] = { truncation };
    at::Tensor truncationTensor = torch::from_blob(truncation_data, at::IntList({ 1 }), at::TensorOptions(at::kFloat)).to(*mDevice.get());

    //
    // Forward process
    // output_tensor { batch, channel, height, width }
    //
    at::Tensor outputTensor = mModule.forward({ mNoiseTensor, classTensor, truncationTensor }).toTensor();

    // Denormalize
    BigGAN::denormalize_(outputTensor);

    // outputTensor { height, width, channel }
    outputTensor = outputTensor.squeeze().permute({ 1, 2, 0 }).to(torch::kCPU);

    float* data = outputTensor.data<float>();

    img.setFromPixels(data, outputTensor.size(1), outputTensor.size(0), OF_IMAGE_COLOR, false);
}

void BigGAN::stepClass()
{
    if (mClassStepCounter > mClassMaxSteps)
    {
        mClassStepCounter = 0.0f;
        mStartClassTensor = mGoalClassTensor;
        mGoalClassTensor = oneHotFromInt(static_cast<int>(ofRandom(mNumClasses)));
    }

    at::Tensor diff = mGoalClassTensor.sub(mStartClassTensor);
    diff.mul_(mClassStepCounter / mClassMaxSteps);
    mClassTensor = mStartClassTensor.add(diff);

    mClassStepCounter += 1.0f;
}

void BigGAN::stepNoise(const float truncation)
{
    if (mNoiseStepCounter > mNoiseMaxSteps)
    {
        mNoiseStepCounter = 0.0f;
        mStartNoiseTensor = mGoalNoiseTensor;
        mGoalNoiseTensor = truncatedNoiseSample({ mBatchSize, mDimZ }, truncation);
    }

    at::Tensor diff = mGoalNoiseTensor.sub(mStartNoiseTensor);
    diff.mul_(mNoiseStepCounter / mNoiseMaxSteps);
    mNoiseTensor = mStartNoiseTensor.add(diff);

    mNoiseStepCounter += 1.0f;
}

void BigGAN::randomizeNoise(const float truncation)
{
    mNoiseTensor = truncatedNoiseSample({ mBatchSize, mDimZ }, truncation);
}

at::Tensor BigGAN::truncatedNoiseSample(c10::IntArrayRef size, const float truncation)
{
    at::Tensor noise = torch::zeros(size, at::TensorOptions(at::kFloat)).to(*mDevice.get());

    at::Tensor tmp = torch::empty({size[0], size[1], 4}, at::TensorOptions(at::kFloat)).to(*mDevice.get()).normal_();
    at::Tensor valid = (tmp < 2) * (tmp > -2);
    at::Tensor ind = std::get<1>(valid.max(-1, true));
    noise.copy_(tmp.gather(-1, ind).squeeze(-1));
    //noise.mul_(1.0).add_(0.0);
    return noise.mul_(truncation);
}

at::Tensor BigGAN::oneHotFromInt(const int index)
{
    float oneHot[mNumClasses] = { 0.0f };
    for (int i = 0; i < mNumClasses; i++)
    {
        if (i == index) oneHot[index] = 1.0f;
    }
    return torch::from_blob(oneHot, at::IntList({ mBatchSize, mNumClasses }), at::TensorOptions(at::kFloat)).to(*mDevice.get());
}

//--------------------------------------------------------------
// Private
//--------------------------------------------------------------