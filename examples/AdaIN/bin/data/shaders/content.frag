#version 450 core
precision mediump float;

in vec4 vPosition;
in vec2 vTexCoord;

uniform vec2 uResolution;
uniform float uTime;

out vec4 fragColor;

/* ~~~ Util: Ashima Webgl Noise ~~~ */

// Description : Array and textureless GLSL 2D simplex noise function.
//      Author : Ian McEwan, Ashima Arts.
//  Maintainer : ijm
//     Lastmod : 20110822 (ijm)
//     License : Copyright (C) 2011 Ashima Arts. All rights reserved.
//               Distributed under the MIT License. See LICENSE file.
//               https://github.com/ashima/webgl-noise
// 

vec3 mod289(vec3 x) {
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec2 mod289(vec2 x) {
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec3 permute(vec3 x) {
  return mod289(((x*34.0)+1.0)*x);
}

float snoise(vec2 v)
  {
  const vec4 C = vec4(0.211324865405187,  // (3.0-sqrt(3.0))/6.0
                      0.366025403784439,  // 0.5*(sqrt(3.0)-1.0)
                     -0.577350269189626,  // -1.0 + 2.0 * C.x
                      0.024390243902439); // 1.0 / 41.0
// First corner
  vec2 i  = floor(v + dot(v, C.yy) );
  vec2 x0 = v -   i + dot(i, C.xx);

// Other corners
  vec2 i1;
  //i1.x = step( x0.y, x0.x ); // x0.x > x0.y ? 1.0 : 0.0
  //i1.y = 1.0 - i1.x;
  i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
  // x0 = x0 - 0.0 + 0.0 * C.xx ;
  // x1 = x0 - i1 + 1.0 * C.xx ;
  // x2 = x0 - 1.0 + 2.0 * C.xx ;
  vec4 x12 = x0.xyxy + C.xxzz;
  x12.xy -= i1;

// Permutations
  i = mod289(i); // Avoid truncation effects in permutation
  vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 ))
		+ i.x + vec3(0.0, i1.x, 1.0 ));

  vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
  m = m*m ;
  m = m*m ;

// Gradients: 41 points uniformly over a line, mapped onto a diamond.
// The ring size 17*17 = 289 is close to a multiple of 41 (41*7 = 287)

  vec3 x = 2.0 * fract(p * C.www) - 1.0;
  vec3 h = abs(x) - 0.5;
  vec3 ox = floor(x + 0.5);
  vec3 a0 = x - ox;

// Normalise gradients implicitly by scaling m
// Approximation of: m *= inversesqrt( a0*a0 + h*h );
  m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );

// Compute final noise value at P
  vec3 g;
  g.x  = a0.x  * x0.x  + h.x  * x0.y;
  g.yz = a0.yz * x12.xz + h.yz * x12.yw;
  return 130.0 * dot(m, g);
}

/* ~~~ End Webgl Noise ~~~ */


#define NB_OCTAVES 6
#define LACUNARITY 2.0
#define GAIN 0.5

float fbm(in vec2 p) {
    float total = 0.0,
          frequency = 1.0,
          amplitude = 1.0;
    
    for (int i = 0; i < NB_OCTAVES; i++) {
        total += snoise(p * frequency) * amplitude;
        frequency *= LACUNARITY;
        amplitude *= GAIN;
    }    
    return total;
}

float t;
float c1, c2, s1, s2;

float pattern(in vec2 p, out float c) {
    float t = uTime;
    vec2 q = vec2( fbm( p + vec2(0.0, 0.0) ),
                  fbm(  p + vec2(c2*.1, t*.02)) );

    c = fbm( p + 2.0*q + vec2(c1+c2,-t*.01));
    return fbm( p + 2.0*q );
}


void main()
{
	vec2 uv = vPosition.xy / uResolution.xy;
    t = uTime;
    
    c1 = 0.1 + cos(t * 0.01) * 0.9; c2 = 0.4 + cos(t * 0.06) * 0.4;
    s1 = sin(t * 0.03); s2 = sin(t*0.3);
    
    vec2 p = vec2(uv.x + c1 * 0.4 + c2 * 0.6, uv.y * 0.3 );

    p.x *= 0.4+ c2 * 0.4;

    float c;
    vec3 col = vec3(pattern(p, c));
    col.r = .6+mix(col.x, c, .2);
    col.b = 0.2+mix(col.x, c, .5)*.1;

	fragColor = vec4(col,1.0);
}