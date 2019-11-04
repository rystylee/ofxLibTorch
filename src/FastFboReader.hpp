// ----------------------------------------------------
// Based on ofxFastFboReader by @satoruhiga
// https://github.com/satoruhiga/ofxFastFboReader
// ----------------------------------------------------

#pragma once

#include "ofMain.h"

namespace ofx {
namespace libtorch {

class FastFboReader
{
public:
    FastFboReader(const int num_buffers = 5);
    ~FastFboReader();

    bool readToPixels(const ofFbo &fbo, ofPixelsRef pix, ofImageType type = OF_IMAGE_COLOR);
    bool readToFloatPixels(const ofFbo &fbo, ofFloatPixelsRef pix, ofImageType type = OF_IMAGE_COLOR);

    bool getAsync() { return async; }
    void setAsync(bool v) { async = v; }

protected:
    const int num_buffers;

    GLuint *pboIds;
    int index, nextIndex;
    size_t num_bytes;
    bool async;

    FastFboReader(const FastFboReader&);
    FastFboReader& operator=(const FastFboReader&);

    void genPBOs();
    void setupPBOs(int num_bytes);

};

} // namespace libtorch
} // namespace ofx
