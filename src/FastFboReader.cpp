#include "FastFboReader.hpp"

namespace ofx {
namespace libtorch {

FastFboReader::FastFboReader(const int num_buffers)
    : pboIds(NULL)
    , async(true)
    , index(0)
    , nextIndex(0)
    , num_bytes(0)
    , num_buffers(num_buffers)
{}

FastFboReader::~FastFboReader()
{
    if (pboIds != NULL)
    {
        glDeleteBuffers(num_buffers, pboIds);
        delete [] pboIds;
        pboIds = NULL;
    }
}

bool FastFboReader::readToPixels(const ofFbo &fbo, ofPixelsRef pix, ofImageType type)
{
    genPBOs();

    int channels;
    int glType;
    
    if (type == OF_IMAGE_COLOR)
    {
        channels = 3;
        glType = GL_RGB;
    }
    else if (type == OF_IMAGE_COLOR_ALPHA)
    {
        channels = 4;
        glType = GL_RGBA;
    }
    else if (type == OF_IMAGE_GRAYSCALE)
    {
        channels = 1;
        glType = GL_LUMINANCE;
    }
    else
    {
        return false;
    }

    const int width = fbo.getWidth();
    const int height = fbo.getHeight();

    if (async)
    {
        index = (index + 1) % num_buffers;
        nextIndex = (index + 1) % num_buffers;
    }
    else
    {
        index = nextIndex = 0;
    }

    size_t nb = width * height * channels;

    if (nb != num_bytes)
    {
        num_bytes = nb;
        setupPBOs(num_bytes);
    }

    glReadBuffer(GL_FRONT);

    fbo.bind();

    glBindBuffer(GL_PIXEL_PACK_BUFFER, pboIds[index]);
    glReadPixels(0, 0, width, height, glType, GL_UNSIGNED_BYTE, NULL);

    glBindBuffer(GL_PIXEL_PACK_BUFFER, pboIds[nextIndex]);
    unsigned char* mem = (unsigned char*)glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);

    if (mem)
    {
        pix.setFromPixels(mem, width, height, channels);
        glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
    }

    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

    fbo.unbind();
    
    return mem != NULL;
}

bool FastFboReader::readToFloatPixels(const ofFbo &fbo, ofFloatPixelsRef pix, ofImageType type)
{
    genPBOs();
    
    int channels;
    int glType;

    if (type == OF_IMAGE_COLOR)
    {
        channels = 3;
        glType = GL_RGB;
    }
    else if (type == OF_IMAGE_COLOR_ALPHA)
    {
        channels = 4;
        glType = GL_RGBA;
    }
    else if (type == OF_IMAGE_GRAYSCALE)
    {
        channels = 1;
        glType = GL_LUMINANCE;
    }
    else
    {
        return false;
    }

    const int width = fbo.getWidth();
    const int height = fbo.getHeight();

    if (async)
    {
        index = (index + 1) % num_buffers;
        nextIndex = (index + 1) % num_buffers;
    }
    else
    {
        index = nextIndex = 0;
    }
    
    size_t nb = width * height * channels * sizeof(float);

    if (nb != num_bytes)
    {
        num_bytes = nb;
        setupPBOs(num_bytes);
    }

    glReadBuffer(GL_FRONT);

    fbo.bind();

    glBindBuffer(GL_PIXEL_PACK_BUFFER, pboIds[index]);
    glReadPixels(0, 0, width, height, glType, GL_FLOAT, NULL);

    glBindBuffer(GL_PIXEL_PACK_BUFFER, pboIds[nextIndex]);
    float* mem = (float*)glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);

    if (mem)
    {
        pix.setFromPixels(mem, width, height, channels);
        glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
    }

    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

    fbo.unbind();

    return mem != NULL;
}

void FastFboReader::genPBOs()
{
    if (!pboIds)
    {
        pboIds = new GLuint[num_buffers];
        glGenBuffers(num_buffers, pboIds);
    }
}

void FastFboReader::setupPBOs(int num_bytes)
{
    for (int i = 0; i < num_buffers; i++)
    {
        glBindBuffer(GL_PIXEL_PACK_BUFFER, pboIds[i]);
        glBufferData(GL_PIXEL_PACK_BUFFER, num_bytes, NULL, GL_STREAM_READ);
    }
    
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
}

} // namespace libtorch
} // namespace ofx
