#version 330

varying vec2 vTexCoord;

uniform int uIndex;
uniform float uStep;
uniform vec2 uResolution;
uniform float uFrequencyX;
uniform float uFrequencyY;
uniform float uAmplitude;
uniform float uOffset;
uniform float uNoiseInfluence;
uniform float uNoiseScale;
uniform float uNoiseInfluence2;
uniform float uNoiseScale2;
uniform float uOverallScale;
uniform vec3 uColors[8];
uniform float uThresholds[7];

layout(location = 0) out float fragY;
layout(location = 1) out float fragU;
layout(location = 2) out float fragV;

// Perlin noise implementation for GLSL
vec4 permute(vec4 x) {
    return mod(((x*34.0)+1.0)*x, 289.0);
}

vec2 fade(vec2 t) {
    return t*t*t*(t*(t*6.0-15.0)+10.0);
}

float cnoise(vec2 P) {
    vec4 Pi = floor(P.xyxy) + vec4(0.0, 0.0, 1.0, 1.0);
    vec4 Pf = fract(P.xyxy) - vec4(0.0, 0.0, 1.0, 1.0);
    Pi = mod(Pi, 289.0);
    vec4 ix = Pi.xzxz;
    vec4 iy = Pi.yyww;
    vec4 fx = Pf.xzxz;
    vec4 fy = Pf.yyww;
    vec4 i = permute(permute(ix) + iy);
    vec4 gx = 2.0 * fract(i * 0.0243902439) - 1.0;
    vec4 gy = abs(gx) - 0.5;
    vec4 tx = floor(gx + 0.5);
    gx = gx - tx;
    vec2 g00 = vec2(gx.x,gy.x);
    vec2 g10 = vec2(gx.y,gy.y);
    vec2 g01 = vec2(gx.z,gy.z);
    vec2 g11 = vec2(gx.w,gy.w);
    vec4 norm = 1.79284291400159 - 0.85373472095314 * vec4(dot(g00, g00), dot(g01, g01), dot(g10, g10), dot(g11, g11));
    g00 *= norm.x;
    g01 *= norm.y;
    g10 *= norm.z;
    g11 *= norm.w;
    float n00 = dot(g00, vec2(fx.x, fy.x));
    float n10 = dot(g10, vec2(fx.y, fy.y));
    float n01 = dot(g01, vec2(fx.z, fy.z));
    float n11 = dot(g11, vec2(fx.w, fy.w));
    vec2 fade_xy = fade(Pf.xy);
    vec2 n_x = mix(vec2(n00, n01), vec2(n10, n11), fade_xy.x);
    float n_xy = mix(n_x.x, n_x.y, fade_xy.y);
    return 2.3 * n_xy;
}

vec3 rgbToYuvBT709PC(vec3 rgb) {
    float y =  0.2126 * rgb.r + 0.7152 * rgb.g + 0.0722 * rgb.b;
    float u = -0.1146 * rgb.r - 0.3854 * rgb.g + 0.5000 * rgb.b + 0.5;
    float v =  0.5000 * rgb.r - 0.4542 * rgb.g - 0.0458 * rgb.b + 0.5;
    return vec3(y, u, v);
}

vec3 rgbToYuvBT709TV(vec3 rgb) {
    // Oblicz YUV w pełnym zakresie
    float y =  0.2126 * rgb.r + 0.7152 * rgb.g + 0.0722 * rgb.b;
    float u = -0.1146 * rgb.r - 0.3854 * rgb.g + 0.5000 * rgb.b;
    float v =  0.5000 * rgb.r - 0.4542 * rgb.g - 0.0458 * rgb.b;

    // Przeskaluj do ograniczonego zakresu TV (BT.709):
    y = y * (219.0 / 255.0) + (16.0 / 255.0);     // Y: [16, 235]
    u = u * (224.0 / 255.0) + 0.5;                // U: [16, 240]
    v = v * (224.0 / 255.0) + 0.5;                // V: [16, 240]

    return vec3(y, u, v);
}

void main() {
    vec2 st = vTexCoord;
    
    // Apply overall scale
    st /= uOverallScale;
    
    // Calculate Perlin noise values for coordinate distortion
    float noiseVal1 = cnoise(st * uNoiseScale);
    float noiseVal2 = cnoise(st * uNoiseScale2 + vec2(1000.0));
    
    // Distort coordinates using Perlin noise
    vec2 distortedSt = st + vec2(noiseVal1 * uNoiseInfluence, noiseVal2 * uNoiseInfluence2);
    
    // Main wave function using distorted coordinates
    float time = float(uIndex) * uStep;
    float waveValue = sin(distortedSt.x * uFrequencyX + distortedSt.y * uFrequencyY + uOffset + time);
    
    // Normalize waveValue from [-1, 1] to [0, 1] for color mapping
    waveValue = (waveValue + 1.0) / 2.0;

    // Color mapping based on thresholds
    // NOTE: Treat colors as indexes, then we will replace those wihn own color palete
    int color_idx = 0;
    for (int i = 0; i < 7; i++) {
        if (waveValue >= uThresholds[i]) {
            color_idx = i + 1;
        } else {
            break;
        }
    }
    
    vec3 yuv = rgbToYuvBT709TV(uColors[color_idx]);

    fragY = yuv.x;
    fragU = yuv.y;
    fragV = yuv.z;
}
