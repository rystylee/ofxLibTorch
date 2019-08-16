#pragma once

#include "ofMain.h"

#include "ofxLibTorch.h"
#include "ofxImGui.h"

class ofApp : public ofBaseApp
{
public:
    void setup();
    void update();
    void draw();
    void keyPressed(int key);

    void renderGui();

    ofVideoGrabber mVideoGrabber;
    ofFbo mCamFbo;

    ofxLibTorch::VGG16 mVGG16;

    ofxImGui::Gui mGui;
    bool mIsShowGui { true };
};