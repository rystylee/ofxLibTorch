#include "AdaIN.hpp"

namespace ofxLibTorch
{

    AdaIN::AdaIN()
    {
        if (torch::cuda::is_available())
            mDevice = std::make_unique<torch::Device>(torch::kCUDA);
        else
            mDevice = std::make_unique<torch::Device>(torch::kCPU);

        float std_data[] = { 0.229, 0.224, 0.225 };
        mStdTensor =  torch::from_blob(std_data, { 3, 1, 1 }, at::TensorOptions(at::kFloat)).to(*mDevice.get());

        float mean_data[] = { 0.485, 0.456, 0.406 };
        mMeanTensor = torch::from_blob(std_data, { 3, 1, 1 }, at::TensorOptions(at::kFloat)).to(*mDevice.get());
    }

    void AdaIN::init(const std::string& modelPath)
    {
        mModule = torch::jit::load(modelPath, *mDevice.get());
        mModule->eval();

        if (mModule == nullptr)
            std::cerr << "Error: Failed to load Torch Script Module..." << std::endl;
        else
            std::cout << "Succeed to load Torch Script Module!" << std::endl;
    }

    void AdaIN::forward(const ofFbo& contentFbo, const ofFbo& styleFbo, const float alpha)
    {
        torch::NoGradGuard noGrad;

        // Content image
        ofFloatPixels contentPix;
        contentFbo.readToPixels(contentPix);
        //mFastFboReader.readToFloatPixels(contentFbo, contentPix);
        cv::Mat contentMat = util::toCv(contentPix);
        at::Tensor content_tensor = torch::from_blob(contentMat.data, at::IntList({ contentMat.rows, contentMat.cols, contentMat.channels() }), at::TensorOptions(at::kFloat)).to(*mDevice.get());
        content_tensor = content_tensor.permute({ 2, 0, 1 }).unsqueeze(0);

        // Style image
        ofFloatPixels stylePix;
        styleFbo.readToPixels(stylePix);
        //mFastFboReader.readToFloatPixels(styleFbo, stylePix);
        cv::Mat styleMat = util::toCv(stylePix);
        at::Tensor style_tensor = torch::from_blob(styleMat.data, at::IntList({ styleMat.rows, styleMat.cols, styleMat.channels() }), at::TensorOptions(at::kFloat)).to(*mDevice.get());
        style_tensor = style_tensor.permute({ 2, 0, 1 }).unsqueeze(0);

        // Normalize
        normalize_(content_tensor);
        normalize_(style_tensor);

        // Alpha factor
        float alpha_data[] = { alpha };
        at::Tensor alpha_tensor = torch::from_blob(alpha_data, at::IntList({ 1 }), at::TensorOptions(at::kFloat)).to(*mDevice.get());

        //
        // Forward process
        // output_tensor { batch, channel, height, width }
        //
        at::Tensor output_tensor = mModule->forward({ content_tensor, style_tensor, alpha_tensor }).toTensor();

        denormalize_(output_tensor);

        // output_tensor { height, width, channel }
        output_tensor = output_tensor.squeeze().permute({ 1, 2, 0 }).to(torch::kCPU);

        float* data = output_tensor.data<float>();

        ofFloatImage img;
        img.setFromPixels(data, output_tensor.size(1), output_tensor.size(0), OF_IMAGE_COLOR, false);
        img.draw(glm::vec2(0), ofGetWidth(), ofGetHeight());
    }

    void AdaIN::normalize_(at::Tensor& tensor)
    {
        tensor.sub_(mMeanTensor).div_(mStdTensor);
    }

    void AdaIN::denormalize_(at::Tensor& tensor)
    {
        torch::clamp_(tensor.mul_(mStdTensor).add_(mMeanTensor), 0, 1);
    }

} // namespace ofxLibTorch