#include "VGG16.hpp"

namespace ofx {
namespace libtorch {

//--------------------------------------------------------------
// Public
//--------------------------------------------------------------
VGG16::VGG16()
{
    float std_data[] = { 0.229, 0.224, 0.225 };
    mStdTensor =  torch::from_blob(std_data, { 3, 1, 1 }, at::TensorOptions(at::kFloat)).to(*mDevice.get());

    float mean_data[] = { 0.485, 0.456, 0.406 };
    mMeanTensor = torch::from_blob(std_data, { 3, 1, 1 }, at::TensorOptions(at::kFloat)).to(*mDevice.get());

    loadJson("jsons/imagenet_class_index.json");
}

void VGG16::init(const std::string& modelPath)
{
    loadModel(modelPath);
}

//std::pair<int, float> VGG16::forward(const ofFbo& inputFbo)
//{
//    torch::NoGradGuard noGrad;
//
//    at::Tensor inputTensor = util::fboToFloatTensor(inputFbo, *mDevice.get());
//
//    preprocess_(inputTensor);
//
//    //
//    // Forward process
//    // outputTensor { batch, 1000 }
//    //
//    at::Tensor outputTensor = mModule->forward({ inputTensor }).toTensor();
//
//    at::Tensor out = torch::softmax(outputTensor, 1);
//    auto maxId = torch::argmax(out).item();
//    auto maxProb = torch::max(out).item();
//
//    std::cout << "Probability: " << maxProb << std::endl;
//    std::cout << "Label: " << mClassLabels[maxId.toInt()] << std::endl;
//
//    return std::make_pair(maxId.toInt(), maxProb.toFloat());
//}

//--------------------------------------------------------------
// Private
//--------------------------------------------------------------
void VGG16::loadJson(const std::string& filePath)
{
    ofFile file(filePath);
    if (file.exists())
    {
        ofJson json = ofLoadJson(filePath);
        for (int i = 0; i < 1000; i++)
        {
            mClassLabels[i] = json[ofToString(i)][1].dump();
        }
    }
    else
    {
        std::cerr << "Error: file not found JSON file: " << filePath << "..." << std::endl;
    }
}

} // namespace liborh
} // namespace ofx