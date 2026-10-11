// about: a living pattern that grows from scattered seeds into spots, worms and mazes, like coral or lichen spreading over the wall (reaction and diffusion). It builds on its own last frames; stays inside the mask. Gives a depth.
// type: time
uniform float feed;       // 0.037 0.01 0.08 what the pattern is fed: low = worms and mazes, high = spots that divide
uniform float kill;       // 0.06 0.045 0.07 how fast it dies back: low = it fills the wall, high = thin and sparse
uniform float size;       // 2 1 6       how coarse the pattern is (also how fast it spreads)
uniform float seeds;      // 0.5 0.02 1  how many seeds it starts from
uniform float warm;       // 0.6 0 1
uniform vec3 colour_a;    // #a187b7   the cool colour
uniform vec3 colour_b;    // #f18a58   the warm colour
// one step of the two substances (Gray-Scott): x is eaten where y is, y grows on it and dies back
vec4 grow(sampler2D last, sampler2D specks, vec2 fragCoord) {
    float s = floor(clamp(size, 1.0, 6.0) + 0.5);     // the pattern lives on cells of s x s pixels
    vec2 cell = floor(fragCoord / s), px = 1.0 / iResolution.xy, uv = (cell + 0.5) * s * px, d = s * px;
    float inside = kMask(uv);
    if (iFrame < 1) {                                 // seeds: a few blocks of 3 x 3 cells
        float n = texture(specks, (floor(cell / 3.0) + 0.5) / 256.0).r;
        return vec4(1.0, step(1.0 - 0.03 * seeds, n) * inside, 0.0, 1.0);
    }
    vec2 c = texture(last, uv).xy;
    vec2 lap = -c + 0.2 * (texture(last, uv + vec2(d.x, 0.0)).xy + texture(last, uv - vec2(d.x, 0.0)).xy
                           + texture(last, uv + vec2(0.0, d.y)).xy + texture(last, uv - vec2(0.0, d.y)).xy)
                  + 0.05 * (texture(last, uv + d).xy + texture(last, uv - d).xy
                            + texture(last, uv + vec2(d.x, -d.y)).xy + texture(last, uv + vec2(-d.x, d.y)).xy);
    float r = c.x * c.y * c.y;
    c += vec2(lap.x - r + feed * (1.0 - c.x), 0.5 * lap.y + r - (kill + feed) * c.y);
    return vec4(clamp(c, 0.0, 1.0) * vec2(1.0, inside), 0.0, 1.0);
}
// four steps per frame: each tab takes the one before it one step further
// Buffer A: 0 = D, 1 = noise
void mainImage(out vec4 fragColor, in vec2 fragCoord) { fragColor = grow(iChannel0, iChannel1, fragCoord); }
// Buffer B: 0 = A, 1 = noise
void mainImage(out vec4 fragColor, in vec2 fragCoord) { fragColor = iFrame < 1 ? texture(iChannel0, fragCoord / iResolution.xy) : grow(iChannel0, iChannel1, fragCoord); }
// Buffer C: 0 = B, 1 = noise
void mainImage(out vec4 fragColor, in vec2 fragCoord) { fragColor = iFrame < 1 ? texture(iChannel0, fragCoord / iResolution.xy) : grow(iChannel0, iChannel1, fragCoord); }
// Buffer D: 0 = C, 1 = noise
void mainImage(out vec4 fragColor, in vec2 fragCoord) { fragColor = iFrame < 1 ? texture(iChannel0, fragCoord / iResolution.xy) : grow(iChannel0, iChannel1, fragCoord); }
// Image: 0 = D
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    vec2 h = 0.5 * floor(clamp(size, 1.0, 6.0) + 0.5) / iResolution.xy;      // between the cells: round outlines
    float v = 0.25 * (texture(iChannel0, uv + h).y + texture(iChannel0, uv - h).y + texture(iChannel0, uv + vec2(h.x, -h.y)).y + texture(iChannel0, uv + vec2(-h.x, h.y)).y);
    v = smoothstep(0.04, 0.34, v);
    vec3 col = mix(vec3(0.07, 0.06, 0.09), mix(colour_a, colour_b, warm), v) + vec3(1.0, 0.95, 0.9) * smoothstep(0.75, 1.0, v) * 0.25;
    kDepth = v;
    fragColor = vec4(col * kMask(uv), 1.0);
}
