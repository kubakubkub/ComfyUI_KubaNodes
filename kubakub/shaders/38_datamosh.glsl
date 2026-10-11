// about: datamosh: the clip on image0 breaks into blocks that smear along its own shapes until a clean frame snaps back. Needs a clip (a moving pattern without one).
// type: time
// optional: image0
uniform float blocks;     // 48 6 240    blocks across the picture
uniform float smear;      // 6 0 40      how far a block drags per frame, in pixels
uniform float keys;       // 3 1 24      clean frames in one loop
uniform float heal;       // 0.02 0 0.5  how many blocks refresh by themselves each frame
uniform float bleed;      // 0.25 0 1    colours drifting apart while it smears
float hash1(vec2 p) {
    vec3 q = fract(vec3(p.xyx) * 0.1031);
    q += dot(q, q.yzx + 33.33);
    return fract((q.x + q.y) * q.z);
}
uniform vec3 colour_a;    // #a187b7   the cool colour
uniform vec3 colour_b;    // #f18a58   the warm colour
const float TAU = 6.2831853;
vec3 src(vec2 uv) {
    if (iChannelResolution[0].x > 1.5) return texture(iChannel0, clamp(uv, 0.0, 1.0)).rgb;
    vec2 p = (uv - 0.5) * vec2(iResolution.x / iResolution.y, 1.0);
    float a = TAU * kLoop;
    vec2 c = vec2(0.45 * cos(a), 0.25 * sin(a * 2.0));
    float ring = smoothstep(0.25, 0.2, abs(fract(length(p - c) * 5.0 - kLoop * 4.0) - 0.5));
    float bar = smoothstep(0.06, 0.04, abs(p.y - 0.3 * sin(a * 3.0 + p.x * 4.0)));
    return mix(colour_a, colour_b, 0.5 + 0.5 * sin(a + p.x * 5.0)) * ring * smoothstep(0.5, 0.1, length(p - c)) + vec3(1.0, 0.9, 0.8) * bar;
}

float lumOf(vec2 uv) { return dot(src(uv), vec3(0.299, 0.587, 0.114)); }
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = kUV;
    vec2 grid = vec2(blocks, floor(blocks * iResolution.y / iResolution.x + 0.5));
    vec2 id = floor(uv * grid), centre = (id + 0.5) / grid, e = 1.0 / grid;
    vec2 g = vec2(lumOf(centre + vec2(e.x, 0.0)) - lumOf(centre - vec2(e.x, 0.0)), lumOf(centre + vec2(0.0, e.y)) - lumOf(centre - vec2(0.0, e.y)));
    vec2 dir = length(g) > 1e-3 ? normalize(g) : vec2(hash1(id) - 0.5, hash1(id + 9.1) - 0.5) * 2.0;
    vec2 mv = dir * smear / iResolution.xy * (0.4 + hash1(id + 3.3));
    float frames = max(1.0, kSeconds * iFrameRate);
    bool clean = iFrame == 0 || fract(kLoop * max(1.0, floor(keys))) < max(1.0, floor(keys)) / frames;
    bool healed = hash1(id + float(iFrame) * 0.37) < heal;
    vec3 now = src(uv);
    vec3 old = vec3(kPrev(uv - mv * (1.0 + bleed)).r, kPrev(uv - mv).g, kPrev(uv - mv * (1.0 - bleed)).b);
    fragColor = vec4((clean || healed ? now : old) * kMask(uv), 1.0);
}
