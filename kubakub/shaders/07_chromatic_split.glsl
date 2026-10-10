// about: the picture on image0 with its colours pulled apart; a wave runs through and the sound pushes it.
uniform float amount;     // 0.012 0 0.1  how far red and blue move apart
uniform float wave;       // 0.004 0 0.05 a sideways wobble that runs down the picture
uniform float lines;      // 40 1 400     how many wobbles from top to bottom
uniform float sound;      // 1 0 4        the level of the sound adds to the split

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = kUV;
    float a = 6.2831853 * kLoop;
    uv.x += wave * sin(uv.y * lines + a * 2.0);
    vec2 dir = normalize(uv - 0.5 + 1e-5);
    float k = amount * (1.0 + sound * kLevel) * (0.3 + length(uv - 0.5) * 1.4);
    float r = texture(iChannel0, clamp(uv + dir * k, 0.0, 1.0)).r;
    vec4 g = texture(iChannel0, clamp(uv, 0.0, 1.0));
    float b = texture(iChannel0, clamp(uv - dir * k, 0.0, 1.0)).b;
    fragColor = vec4(vec3(r, g.g, b) * kMask(kUV), 1.0);
}
