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
    ofxLibTorch::BicycleGAN mBicycleGAN;

    ofFloatImage mImg;

    ofxImGui::Gui mGui;
    bool mIsVisibleGui { true };
};
