// about: the picture on image0 in two colours of your choice: one for the shadows, one for the lights, like a risograph or a poster. The loop can swing the balance.
// type: lens
uniform vec3 shadows;     // #1c1433   the colour of the dark parts
uniform vec3 lights;      // #f18a58   the colour of the bright parts
uniform float contrast;   // 1.3 0.2 4
uniform float balance;    // 0 -0.5 0.5    more of the shadow colour, or more of the light one
uniform float amount;     // 1 0 1         0 = the picture as it is
uniform float swing;      // 0 0 0.5       how far the balance swings within one loop
uniform float sound;      // 0 0 1         the low end of a sound pushes it to the lights
const float TAU = 6.2831853;

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = kUV;
    vec3 c = texture(iChannel0, uv).rgb;
    float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
    l = clamp((l - 0.5) * contrast + 0.5 + balance + swing * sin(TAU * kLoop) + sound * kLow * 0.4, 0.0, 1.0);
    fragColor = vec4(mix(c, mix(shadows, lights, smoothstep(0.0, 1.0, l)), amount) * kMask(uv), 1.0);
}
