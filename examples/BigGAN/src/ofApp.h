#pragma once

#include "ofMain.h"

#include "BigGAN.hpp"
#include "ofxImGui.h"

class ofApp : public ofBaseApp
{
public:
    void setup();
    void update();
    void draw();
    void keyPressed(int key);

    void renderGui();

    const int mBatchSize { 1 };
    const int mZDim { 128 };

    int mClassIndex { 0 };
    float mTruncation { 0.4f };

    ofFloatImage mImg;

    BigGAN mBigGAN;

    ofxImGui::Gui mGui;
    bool mIsShowGui { true };
};
