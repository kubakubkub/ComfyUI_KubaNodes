// about: the picture on image0 as scan lines that the brightness pushes out of place, like a signal on a tube; the lines drift. Loops without a jump.
// type: signal
// The idea (brightness shifts the phase of a cosine carrier, then steps and smoothing) is olegchomp's td-modulation
// for TouchDesigner (github.com/olegchomp/td-rnd); written new from its description.
uniform float rate;       // 90 10 400   lines down the picture
uniform float depth;      // 5 0 30      how far the brightness pushes a line
uniform float steps;      // 3 1 16      grey steps (1 = hard black and white)
uniform float drift;      // 2 0 8       times the lines run through in one loop
uniform float soften;     // 1 0 4       softness of the lines
uniform float tint;       // 0 0 1       0 = white lines, 1 = in the picture's colours
uniform float light;      // 0.75 0 1    0 = all lines equally bright, 1 = bright only where the picture is
const float TAU = 6.2831853;
vec3 pic(vec2 p) { return texture(iChannel0, clamp(p, 0.0, 1.0)).rgb; }

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = kUV, px = 1.0 / iResolution.xy;
    vec3 c = vec3(0.0);
    for (int k = -2; k <= 2; k++)                     // the signal is soft: a blur about one line wide
        c += (pic(uv + vec2(float(k) * 0.3, 0.2) / rate) + pic(uv + vec2(float(k) * 0.3, -0.2) / rate)) * 0.1;
    float lum = dot(c, vec3(0.299, 0.587, 0.114));
    float n = max(1.0, floor(steps + 0.5)), v = 0.0;
    for (int k = -2; k <= 2; k++) {                   // a few samples down the pixel: the steps get soft edges
        float y = uv.y + float(k) * px.y * 0.5 * soften;
        float wave = 0.5 + 0.5 * cos(TAU * (rate * y + floor(drift + 0.5) * kLoop) + depth * lum);
        v += floor(wave * n + 0.5) / n * 0.2;
    }
    v *= mix(1.0, smoothstep(0.0, 0.7, lum) * 1.25, light);
    vec3 col = mix(vec3(v), c / max(lum, 0.05) * v, tint);
    fragColor = vec4(clamp(col, 0.0, 1.0) * kMask(kUV), 1.0);
}
