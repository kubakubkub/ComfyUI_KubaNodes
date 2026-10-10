// about: a wall of bricks; a wave of light runs through them along the field (or left to right). Loops without a jump.
uniform float rows;       // 14 2 80     rows of bricks
uniform float ratio;      // 2.2 1 5     how long a brick is
uniform float mortar;     // 0.08 0 0.4  the gap between bricks
uniform float width;      // 0.3 0.05 1  how wide the wave is
uniform float waves;      // 1 1 8       how many waves run at once
const vec3 LAV = vec3(0.63, 0.53, 0.72), ORG = vec3(0.95, 0.54, 0.35);
float hash1(vec2 p) {
    vec3 q = fract(vec3(p.xyx) * 0.1031);
    q += dot(q, q.yzx + 33.33);
    return fract((q.x + q.y) * q.z);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 p = fragCoord / iResolution.y * rows;
    p.x /= ratio;
    float row = floor(p.y);
    p.x += mod(row, 2.0) * 0.5;
    vec2 id = vec2(floor(p.x), row), f = fract(p) - 0.5;
    vec2 centre = (vec2((id.x + 0.5 - mod(row, 2.0) * 0.5) * ratio, id.y + 0.5)) / rows * vec2(iResolution.y / iResolution.x, 1.0);
    float along = kField(clamp(centre, 0.0, 1.0));
    float x = fract(along * floor(waves) - kLoop + hash1(id) * 0.08);
    float on = smoothstep(width, 0.0, min(x, 1.0 - x) * 2.0);
    vec2 b = smoothstep(vec2(0.5), vec2(0.5) - vec2(0.04 / ratio, 0.04), abs(f) + vec2(mortar / ratio, mortar) * 0.5);
    vec3 stone = mix(LAV * 0.35, ORG * 0.35, hash1(id + 2.7));
    vec3 col = (stone * 0.5 + mix(ORG, vec3(1.0, 0.92, 0.8), on) * on) * b.x * b.y;
    fragColor = vec4(col * kMask(kUV), 1.0);
}
