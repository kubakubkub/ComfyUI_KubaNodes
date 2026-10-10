// about: the picture on image0 as a worn VHS tape: soft, colours bleeding sideways, a wobbling edge, snow, a noisy band and the torn lines of the head switch at the bottom.
// The steps and their order follow the VHS signal chain of ntsc-rs (github.com/valadaptive/ntsc-rs, valadaptive),
// found through olegchomp's TouchDesigner version (github.com/olegchomp/ntsc-td); written new as one shader pass.
uniform float wear;       // 0.5 0 1     how worn the tape is: everything at once
uniform float bleed;      // 1 0 3       colours smearing sideways and lagging behind
uniform float wave;       // 1 0 4       the wobble of the picture's edge
uniform float snow;       // 1 0 4       white streaks
uniform float tracking;   // 0.14 0 0.6  height of the noisy band at the bottom (0 = none)
uniform float head;       // 0.03 0 0.2  height of the torn lines at the very bottom (0 = none)
uniform float ring;       // 0.7 0 2     bright and dark rims after hard edges
uniform float grain;      // 1 0 4       noise in the light and in the colours
uniform float lines;      // 480 120 1080 lines of the tape
const float TAU = 6.2831853;
float hash1(vec2 p) {
    vec3 q = fract(vec3(p.xyx) * 0.1031);
    q += dot(q, q.yzx + 33.33);
    return fract((q.x + q.y) * q.z);
}
float noise1(float x, float s) {                      // smooth noise along one line of numbers
    float i = floor(x), f = fract(x);
    return mix(hash1(vec2(i, s)), hash1(vec2(i + 1.0, s)), f * f * (3.0 - 2.0 * f));
}
vec3 yiq(vec2 p) {                                    // outside the tape there is no signal
    if (p.x < 0.0 || p.x > 1.0) return vec3(0.0);
    vec3 c = texture(iChannel0, vec2(p.x, clamp(p.y, 0.0, 1.0))).rgb;
    return vec3(dot(c, vec3(0.299, 0.587, 0.114)), dot(c, vec3(0.5959, -0.2746, -0.3213)), dot(c, vec3(0.2115, -0.5227, 0.3112)));
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = kUV;
    float line = floor(uv.y * lines), ly = (line + 0.5) / lines;
    float T = floor(fract(kLoop) * 120.0);            // the noise of this frame; the same again after one loop
    float w = clamp(wear, 0.0, 1.0);

    // the edge wave: every line sits a little to the side, slowly wandering, plus a jitter per line
    float shift = (sin(ly * 9.0 + TAU * kLoop * 2.0) + 0.6 * sin(ly * 23.0 - TAU * kLoop * 3.0) + 0.4 * sin(ly * 61.0 + TAU * kLoop * 5.0))
                  * 0.0012 * wave * (0.3 + w);
    shift += (hash1(vec2(line, T)) - 0.5) * 0.0016 * wave * w;
    // tracking noise: towards the bottom the lines wave harder
    float tb = tracking > 0.0 ? clamp(1.0 - uv.y / tracking, 0.0, 1.0) : 0.0;
    shift += tb * tb * (noise1(line * 0.5 + T * 3.7, 7.0) - 0.5) * 0.09 * (0.4 + w);
    // head switching: the last lines are pushed aside, more the lower they are
    float hb = head > 0.0 ? clamp(1.0 - uv.y / head, 0.0, 1.0) : 0.0;
    shift += pow(hb, 1.5) * 0.16 + step(0.0001, hb) * (hash1(vec2(line, T + 31.0)) - 0.5) * 0.006;

    vec2 p = vec2(uv.x - shift, ly);
    float du = 0.75 / lines;                          // the tape cannot draw anything finer than this
    float Y = 0.0, ws = 0.0;
    for (int k = -4; k <= 4; k++) {                   // the light: a narrow blur along the line
        float g = exp(-float(k * k) / (2.0 + 6.0 * w));
        Y += yiq(p + vec2(float(k) * du * 0.6, 0.0)).x * g;
        ws += g;
    }
    Y /= ws;
    vec3 wide = vec3(0.0);                            // the colour: a wide blur, lagging behind, mixed with the line above
    float cw = 0.0, spread = du * (2.0 + 5.0 * w) * bleed, lag = du * 3.0 * bleed * (0.3 + w);
    for (int k = -5; k <= 5; k++) {
        float g = exp(-float(k * k) / 9.0);
        vec2 o = vec2(float(k) * spread * 0.45 - lag, 0.0);
        wide += (yiq(p + o) + yiq(p + o + vec2(0.0, 1.0 / lines))) * 0.5 * g;
        cw += g;
    }
    wide /= cw;
    float I = wide.y, Q = wide.z;
    Y += (Y - mix(Y, wide.x, 0.8)) * ring * 1.2;       // the recorder sharpens what the tape lost: rims after edges
    Y += (yiq(p - vec2(du * 5.0, 0.0)).x - yiq(p - vec2(du * 8.0, 0.0)).x) * 0.08 * ring;      // a faint echo

    if (hash1(vec2(line, T + 57.0)) < 0.03 * w * w) { I = 0.0; Q = 0.0; }                      // a line that lost its colour

    // snow: short bright streaks that ring out, more of them in the tracking band
    float chance = snow * (0.012 + 0.05 * w * w) + tb * tb * 0.7 * step(0.0001, snow + w);
    for (int j = 0; j < 3; j++) {
        vec2 sd = vec2(line + float(j) * 131.0, T + float(j) * 17.0);
        if (hash1(sd) < chance) {
            float x0 = hash1(sd + 3.3) * 1.1 - 0.1, len = mix(0.012, 0.09, hash1(sd + 7.7));
            float d = (uv.x - x0) / len;
            if (d > 0.0 && d < 1.0) Y += cos(d * 3.14159 * 4.0) * (1.0 - d) * (1.0 - d) * (0.4 + 1.2 * hash1(sd + 9.1));
        }
    }
    // noise in the light (fine) and in the colour (coarse), strong in the tracking band
    float n = hash1(vec2(floor(uv.x * lines * 1.6) + T * 13.0, line)) - 0.5;
    Y += n * (0.07 * grain * (0.3 + w) + tb * tb * 0.9);
    vec2 cn = vec2(hash1(vec2(floor(uv.x * lines * 0.2) + T * 5.0, line + 11.0)), hash1(vec2(floor(uv.x * lines * 0.2) + T * 9.0, line + 47.0))) - 0.5;
    I += cn.x * 0.14 * grain * w;
    Q += cn.y * 0.14 * grain * w;
    I *= 1.0 - 0.6 * tb;                              // the band loses its colour
    Q *= 1.0 - 0.6 * tb;

    Y = Y * (1.0 - 0.07 * w) + 0.035 * w;             // washed out blacks and whites
    float sat = 1.0 - 0.2 * w;
    vec3 col = vec3(Y + 0.956 * I * sat + 0.619 * Q * sat, Y - 0.272 * I * sat - 0.647 * Q * sat, Y - 1.106 * I * sat + 1.703 * Q * sat);
    fragColor = vec4(clamp(col, 0.0, 1.0) * kMask(kUV), 1.0);
}
