// about: jumping digital bars and blocks, a broken signal. Loops; the highs of a sound add more.
// type: pattern
// arrow: right
uniform float rows;       // 24 2 200    rows of bars
uniform float amount;     // 0.35 0 1    how many bars are lit
uniform float steps;      // 16 1 120    how many times the picture changes in one loop
uniform float colour;     // 0.6 0 1     0 = white, 1 = violet and orange
uniform float sound;      // 0 0 2
uniform vec3 colour_a;    // #a187b7   the cool colour
uniform vec3 colour_b;    // #f18a58   the warm colour
float hash1(vec2 p) {
    vec3 q = fract(vec3(p.xyx) * 0.1031);
    q += dot(q, q.yzx + 33.33);
    return fract((q.x + q.y) * q.z);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = kUV;
    float step_ = floor(fract(kLoop) * floor(steps));
    float row = floor(uv.y * rows);
    float h = hash1(vec2(row, step_));
    float cols = floor(mix(2.0, 40.0, hash1(vec2(row, step_ + 9.0))));
    float col_ = floor((uv.x + hash1(vec2(row, step_ + 3.0))) * cols);
    float on = step(1.0 - amount * (1.0 + sound * kHigh), hash1(vec2(col_, row + step_ * 17.0)) * 0.6 + h * 0.4);
    float tall = step(0.92, hash1(vec2(floor(uv.y * rows * 0.2), step_ + 5.0)));
    on = max(on, tall * step(0.5, hash1(vec2(floor(uv.x * 9.0), step_))));
    vec3 tint = mix(vec3(1.0), mix(colour_a, colour_b, step(0.5, hash1(vec2(row + 1.0, col_)))) * 1.3, colour);
    float scan = 0.85 + 0.15 * sin(fragCoord.y * 3.14159);
    fragColor = vec4(tint * on * scan * kMask(uv), 1.0);
}
