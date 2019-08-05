#version 450 core

// -----------------------------
// oF default
// -----------------------------
layout (location = 0) in vec4 position; 
layout (location = 1) in vec4 color; 
layout (location = 2) in vec4 normal; 
layout (location = 3) in vec2 texcoord; 

out vec4 vPosition;
out vec4 vColor;
out vec4 vNormal;
out vec2 vTexCoord;

uniform mat4 modelViewProjectionMatrix;
uniform mat4 modelViewMatrix;

void main()
{
    vPosition = position;
    vColor = color;
    vNormal = normal;
    vTexCoord = texcoord;
    gl_Position = modelViewProjectionMatrix * position;
}