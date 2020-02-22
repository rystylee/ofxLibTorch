#pragma once

#include "ofMain.h"

#include "BicycleGAN.hpp"
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
    BicycleGAN mBicycleGAN;

    ofFloatImage mImg;

    ofxImGui::Gui mGui;
    bool mIsVisibleGui { true };
};
