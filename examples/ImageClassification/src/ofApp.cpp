#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup()
{
    ofSetBackgroundColor(0);
    ofSetVerticalSync(false);

    mVideoGrabber.initGrabber(640, 480);

    ofFboSettings s;
    s.numColorbuffers = 1;
    s.useDepth = false;
    s.useStencil = false;
    s.depthStencilAsTexture = false;
    s.textureTarget = GL_TEXTURE_2D;
    s.internalformat = GL_RGB;
    s.wrapModeHorizontal = GL_CLAMP_TO_EDGE;
    s.wrapModeVertical = GL_CLAMP_TO_EDGE;
    s.minFilter = GL_NEAREST;
    s.maxFilter = GL_NEAREST;
    s.numSamples = 0;
    s.width = 224;
    s.height = 224;
    mCamFbo.allocate(s);

    mVGG16.init("data/models/vgg16.pt");

    mGui.setup();
}

//--------------------------------------------------------------
void ofApp::update()
{
    mVideoGrabber.update();
    if (mVideoGrabber.isFrameNew())
    {
        mCamFbo.begin();
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        mVideoGrabber.draw(glm::vec2(0), mCamFbo.getWidth(), mCamFbo.getHeight());
        mCamFbo.end();

        auto resoult = mVGG16.forward<std::pair<int, at::Tensor>>(mCamFbo);
    }
}

//--------------------------------------------------------------
void ofApp::draw()
{
    mCamFbo.draw(glm::vec2(0), ofGetWidth(), ofGetHeight());

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
        }
        ImGui::End();
    }
    mGui.end();
}

//--------------------------------------------------------------