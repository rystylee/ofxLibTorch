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

    vector<int> mContentDims { 640, 320 };
    vector<int> mStyleDims { 320, 180 };

    ofShader mContentShader, mStyleShader;
    ofFbo mContentFbo, mStyleFbo;
    ofVideoGrabber mVideoGrabber;
    ofVideoPlayer mContentVideoPlayer;
    ofVideoPlayer mStyleVideoPlayer;

    ofxImGui::Gui mGui;
    bool mIsVisibleGui { true };
    bool mIsWebCam { false };
    bool mIsVideoUpdate { true };

    // AdaIN control
    float mAlpha { 1.0f };
};
