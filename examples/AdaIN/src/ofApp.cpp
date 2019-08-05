#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup()
{
    ofSetVerticalSync(false);
    ofSetFrameRate(0);

    mContentShader.load("shaders/pass.vert", "shaders/content.frag");
    mStyleShader.load("shaders/pass.vert", "shaders/style.frag");

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
    s.width = mContentDims[0];
    s.height = mContentDims[1];
    mContentFbo.allocate(s);
    s.width = mStyleDims[0];
    s.height = mStyleDims[1];
    mStyleFbo.allocate(s);

    mVideoGrabber.initGrabber(mContentDims[0], mContentDims[1]);

    mContentVideoPlayer.load("videos/Big_Buck_Bunny.mp4");
    mContentVideoPlayer.setVolume(0);
    mStyleVideoPlayer.load("videos/GHOST_IN_THE_SHELL_-_REEL.mp4");
    mStyleVideoPlayer.setVolume(0);

    mAdaIN.init("data/models/AdaIN_gpu_cw_640_ch_360_sw_320_sh_180.pt");

    mGui.setup();
}

//--------------------------------------------------------------
void ofApp::update()
{
    if (mIsWebCam)
    {
        mVideoGrabber.update();
        if (mVideoGrabber.isFrameNew())
        {
            mContentFbo.begin();
            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            mVideoGrabber.draw(glm::vec2(0), mContentFbo.getWidth(), mContentFbo.getHeight());
            mContentFbo.end();
        }
    }
    else
    {
        mContentFbo.begin();
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        //mContentShader.begin();
        //mContentShader.setUniform2f("uResolution", mContentFbo.getWidth(), mContentFbo.getHeight());
        //mContentShader.setUniform1f("uTime", ofGetElapsedTimef());
        //ofDrawRectangle(glm::vec2(0), mContentFbo.getWidth(), mContentFbo.getHeight());
        //mContentShader.end();
        mContentVideoPlayer.update();
        mContentVideoPlayer.draw(glm::vec2(0), mContentFbo.getWidth(), mContentFbo.getHeight());
        mContentFbo.end();
    }


    mStyleFbo.begin();
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    //mStyleShader.begin();
    //mStyleShader.setUniform2f("uResolution", mStyleFbo.getWidth(), mStyleFbo.getHeight());
    //mStyleShader.setUniform1f("uTime", ofGetElapsedTimef());
    //ofDrawRectangle(glm::vec2(0), mStyleFbo.getWidth(), mStyleFbo.getHeight());
    //mStyleShader.end();
    mStyleVideoPlayer.update();
    mStyleVideoPlayer.draw(glm::vec2(0), mStyleFbo.getWidth(), mStyleFbo.getHeight());
    //mContentVideoPlayer.draw(glm::vec2(0), mStyleFbo.getWidth(), mStyleFbo.getHeight());
    mStyleFbo.end();
}

//--------------------------------------------------------------
void ofApp::draw()
{
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    mAdaIN.forward(mContentFbo, mStyleFbo, mAlpha);

    if (mIsVisibleGui) renderGui();
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

            if (ImGui::CollapsingHeader("AdaIN control"))
            {
                ImGui::Spacing();
                ImGui::SliderFloat("Alpha", &mAlpha, 0.0f, 1.0f);
                ImGui::Checkbox("WebCam", &mIsWebCam);
            }

            ImGui::Spacing();
            if (ImGui::CollapsingHeader("Debug view"))
            {
                ImGui::Spacing();
                ImTextureID id;
                id = reinterpret_cast<ImTextureID>(mContentFbo.getTexture().getTextureData().textureID);
                ImGui::Text("Content: (640, 320)");
                ImGui::Image(id, ImVec2(320, 180));

                ImGui::Spacing();
                id = reinterpret_cast<ImTextureID>(mStyleFbo.getTexture().getTextureData().textureID);
                ImGui::Text("Style: (320, 180)");
                ImGui::Image(id, ImVec2(320, 180));
            }
        }
        ImGui::End();
    }
    mGui.end();
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key)
{
    switch (key)
    {
        case 'g':
        {
            mIsVisibleGui = !mIsVisibleGui;
        }
        case 's':
        {
            mContentVideoPlayer.play();
            mStyleVideoPlayer.play();
        }
        default: break;
    }
}
