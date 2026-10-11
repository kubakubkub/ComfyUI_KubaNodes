// about: the picture on image0 sent through a composite video cable: colour fringes and rainbows crawling along fine detail, dots on colour edges, soft colour.
// type: signal
// Light and colour are put on one signal (the colour on a carrier wave) and taken apart again, as NTSC does it; the
// idea of doing the real thing comes from ntsc-rs (github.com/valadaptive/ntsc-rs, valadaptive), found through
// olegchomp's TouchDesigner version (github.com/olegchomp/ntsc-td); written new as one shader pass.
uniform float carrier;    // 170 40 400  waves of the colour carrier across the picture (lower = coarser fringes)
uniform float fringe;     // 1 0 2       rainbows: colour picked up from fine detail
uniform float dots;       // 0.5 0 1     the dot pattern left in the light at colour edges
uniform float crawl;      // 2 0 8       times the dots crawl through in one loop
uniform float colour;     // 1 0 2       how much colour comes back
uniform float lines;      // 480 120 1080 lines of the picture
const float TAU = 6.2831853;
vec3 yiq(vec2 p) {
    vec3 c = texture(iChannel0, clamp(p, 0.0, 1.0)).rgb;
    return vec3(dot(c, vec3(0.299, 0.587, 0.114)), dot(c, vec3(0.5959, -0.2746, -0.3213)), dot(c, vec3(0.2115, -0.5227, 0.3112)));
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = kUV;
    float line = floor(uv.y * lines), ly = (line + 0.5) / lines;
    float turn = TAU * (0.5 * line + floor(crawl + 0.5) * kLoop);      // the carrier flips from line to line and crawls
    float dx = 1.0 / (carrier * 8.0);                 // eight samples per carrier wave
    float notch = 0.0, near = 0.0;                    // the light: averaged over one wave (the colour cancels), and raw
    vec2 demod = vec2(0.0), clean = vec2(0.0);        // the colour: taken off the carrier, and as it really was
    float ws = 0.0;
    for (int k = -8; k < 8; k++) {
        float x = uv.x + (float(k) + 0.5) * dx;
        vec3 s = yiq(vec2(x, ly));
        float ph = TAU * carrier * x + turn;
        float signal = s.x + s.y * cos(ph) + s.z * sin(ph);            // what travels through the cable
        float g = 0.5 + 0.5 * cos(3.14159 * (float(k) + 0.5) / 8.0);   // a soft window over two waves
        demod += signal * vec2(cos(ph), sin(ph)) * g * 2.0;
        clean += s.yz * g;
        ws += g;
        if (k >= -4 && k < 4) notch += signal * 0.125;
        if (k >= -1 && k < 1) near += signal * 0.5;
    }
    demod /= ws;
    clean /= ws;
    float Y = mix(notch, near, dots * 0.6);
    vec2 iq = mix(clean, demod, fringe) * colour;
    vec3 col = vec3(Y + 0.956 * iq.x + 0.619 * iq.y, Y - 0.272 * iq.x - 0.647 * iq.y, Y - 1.106 * iq.x + 1.703 * iq.y);
    fragColor = vec4(clamp(col, 0.0, 1.0) * kMask(kUV), 1.0);
}
