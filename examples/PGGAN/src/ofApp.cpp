#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup()
{
    ofSetBackgroundColor(0);
    ofSetVerticalSync(false);

    mPGGAN.init("data/models/PGGAN_512.pt");

    mGui.setup();
}

//--------------------------------------------------------------
void ofApp::update()
{
    mPGGAN.stepNoise();
    mPGGAN.forward(mImg);
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

            if (ImGui::CollapsingHeader("PGGAN control"))
            {
                ImGui::Spacing();
            }
        }
        ImGui::End();
    }
    mGui.end();
}

//--------------------------------------------------------------
