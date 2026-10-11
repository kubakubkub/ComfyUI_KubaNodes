// about: the picture on image0 printed in dots, big where it is bright; the raster slowly turns. Loops without a jump.
// type: look
uniform float dots;       // 70 8 300    dots across the picture
uniform float angle;      // 30 0 90     degrees the raster is turned
uniform float turn;       // 1 0 4       quarter turns of the raster in one loop
uniform float colour;     // 1 0 1       0 = one ink colour, 1 = the picture's colours
uniform float soft;       // 0.15 0.02 1
uniform vec3 colour_a;    // #a187b7   the cool colour
uniform vec3 colour_b;    // #f18a58   the warm colour
const float TAU = 6.2831853;
vec3 pic(vec2 uv) { return texture(iChannel0, clamp(uv, 0.0, 1.0)).rgb; }
float lum(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = kUV, asp = vec2(iResolution.x / iResolution.y, 1.0);
    float a = radians(angle) + TAU * 0.25 * floor(turn + 0.5) * kLoop;
    mat2 r = mat2(cos(a), -sin(a), sin(a), cos(a));
    vec2 g = r * ((uv - 0.5) * asp) * dots, id = floor(g), f = fract(g) - 0.5;
    vec2 c = (transpose(r) * ((id + 0.5) / dots)) / asp + 0.5;
    vec3 pc = pic(c);
    float radius = sqrt(clamp(lum(pc), 0.0, 1.0)) * 0.62;
    float v = smoothstep(radius, radius - soft * 0.5, length(f));
    vec3 ink = mix(mix(colour_a, colour_b, lum(pc)), pc / max(lum(pc), 0.05) * 0.8, colour);
    fragColor = vec4(ink * v * kMask(uv), 1.0);
}
