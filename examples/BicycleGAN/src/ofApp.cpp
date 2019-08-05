#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup()
{
    ofSetVerticalSync(false);
    ofSetFrameRate(0);
    ofSetBackgroundColor(0);

    mBicycleGAN.init("data/models/BicycleGAN_gpu.pt");

    mImg.allocate(512, 512, OF_IMAGE_COLOR);

    mGui.setup();
}

//--------------------------------------------------------------
void ofApp::update()
{
    mBicycleGAN.stepNoise();
    mBicycleGAN.forward(mImg);
}

//--------------------------------------------------------------
void ofApp::draw()
{
    mImg.draw(glm::vec2(0), ofGetWidth(), ofGetHeight());

    if (mIsVisibleGui) renderGui();
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key)
{
    switch (key)
    {
        case 'g': mIsVisibleGui = !mIsVisibleGui;
        default: break;
    }
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
        }
        ImGui::End();
    }
    mGui.end();
}

//--------------------------------------------------------------
