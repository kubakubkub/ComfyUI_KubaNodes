// about: stripes that run across at an angle, or along a drawn arrow. Loops without a jump; a beat can flash them.
uniform float stripes;    // 12 1 120    stripes across the picture
uniform float angle;      // 30 -90 90   degrees (a drawn arrow comes first)
uniform float width;      // 0.5 0.02 0.98 how much of each stripe is lit
uniform float soft;       // 0.1 0 1
uniform float speed;      // 2 1 12      stripes that pass in one loop (an even count: 3 runs as 4)
uniform float beat;       // 0 0 1
const vec3 LAV = vec3(0.63, 0.53, 0.72), ORG = vec3(0.95, 0.54, 0.35);

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 p = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    float a = radians(angle);
    float x = dot(p, kVector > 0.5 ? kDir : vec2(cos(a), sin(a))) * stripes - kLoop * 2.0 * floor(speed * 0.5 + 0.5);      // an even count: the two colours meet again
    float d = abs(fract(x) - 0.5) * 2.0;
    float v = 1.0 - smoothstep(width * (1.0 - soft), width + 1e-4, d);
    v *= 1.0 + beat * kBeat;
    vec3 col = mix(LAV, ORG, step(0.5, fract(x * 0.5))) * v;
    fragColor = vec4(col * kMask(kUV), 1.0);
}
