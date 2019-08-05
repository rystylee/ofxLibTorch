#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup()
{
    ofSetBackgroundColor(0);
    ofSetVerticalSync(false);

    mBigGAN.init("data/models/BigGAN_512.pt");

    mGui.setup();
}

//--------------------------------------------------------------
void ofApp::update()
{
    mBigGAN.stepClass();
    mBigGAN.stepNoise(mTruncation);
    mBigGAN.forward(mImg, mTruncation);
}

//--------------------------------------------------------------
void ofApp::draw()
{
    mImg.draw(glm::vec2(0), ofGetWidth(), ofGetHeight());

    if(mIsShowGui) renderGui();
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key)
{
    if (key == 'g') mIsShowGui = !mIsShowGui;
}

//--------------------------------------------------------------
void ofApp::renderGui()
{
    mGui.begin();
    {
        ImGui::Begin("Main");
        {
            ImGui::SetWindowFontScale(1.0);

            ImGui::Text("FPS : %.2f", ofGetFrameRate());
            ImGui::Spacing();

            if (ImGui::CollapsingHeader("BigGAN control"))
            {
                ImGui::Spacing();
                ImGui::SliderInt("Class Index", &mClassIndex, 0, 1000);
                ImGui::SliderFloat("Truncation", &mTruncation, 0.0f, 1.0f);
            }
        }
        ImGui::End();
    }
    mGui.end();
}

//--------------------------------------------------------------
