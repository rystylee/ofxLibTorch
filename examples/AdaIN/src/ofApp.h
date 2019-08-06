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

private:
    ofxLibTorch::AdaIN mAdaIN;

    std::array<int, 2> mContentDims { 640, 320 };
    std::array<int, 2> mStyleDims { 320, 180 };

    ofFbo mContentFbo, mStyleFbo;
    ofVideoPlayer mContentVideoPlayer;
    ofVideoPlayer mStyleVideoPlayer;

    ofxImGui::Gui mGui;
    bool mIsShowGui { true };
    bool mIsVideoUpdate { true };

    // AdaIN control
    float mAlpha { 1.0f };
};
