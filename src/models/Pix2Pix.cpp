#include "Pix2Pix.hpp"

namespace ofxLibTorch
{

    Pix2Pix::Pix2Pix()
    {
    }

    void Pix2Pix::init(const std::string& modelPath, const int inputH, const int inputW)
    {
        mInputH = inputH; mInputW = inputW;

        loadModel(modelPath);

        if (mModule == nullptr)
            std::cerr << "Error: Failed to load Torch Script Module..." << std::endl;
        else
            std::cout << "Succeed to load Torch Script Module!" << std::endl;
    }

    void Pix2Pix::forward(ofFloatImage& img)
    {
        torch::NoGradGuard no_grad;

        cv::Mat inputMat = util::toCv(img.getPixels());
        at::Tensor inputTensor = torch::from_blob(inputMat.data, at::IntList({ inputMat.rows, inputMat.cols, inputMat.channels() }), at::TensorOptions(at::kFloat)).to(*mDevice.get());
        inputTensor = inputTensor.permute({ 2, 0, 1 }).unsqueeze(0);

        // Normalize
        Pix2Pix::normalize_(inputTensor);

        //
        // Forward process
        // outputTensor { batch, channel, height, width }
        //
        at::Tensor outputTensor = mModule->forward({ inputTensor }).toTensor();

        // Denormalize
        Pix2Pix::denormalize_(outputTensor);

        // (c, h, w) -> (h, w, c)
        outputTensor = outputTensor.squeeze().permute({ 1, 2, 0 }).to(torch::kCPU);
        float* data = outputTensor.data<float>();

        img.setFromPixels(data, outputTensor.size(1), outputTensor.size(0), OF_IMAGE_COLOR, false);
    }

    void Pix2Pix::addNoise(ofFloatImage& img, const float strength = 0.1)
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

    void Pix2Pix::invert(ofFloatImage& img)
    {
        auto data = img.getPixels();
        for (auto& d : data)
        {
            d = 1.0 - d;
            d = ofClamp(d, 0.0, 1.0);
        }
        img.setFromPixels(data.getPixels(), img.getWidth(), img.getHeight(), OF_IMAGE_COLOR, true);
    }

} // namespace ofxLibTorch
