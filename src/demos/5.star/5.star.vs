#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in float aEnergy;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform float window_width_half;
uniform float window_height_half;
uniform int pointSize;

out float n_pe;
flat out int window_pos_x;
flat out int window_pos_y;
flat out int ptSize;
//out float grayscale[25];

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0);

    gl_PointSize = pointSize;   // pointSize is determined by sigma
    ptSize = pointSize; // pass to fragment shader
    n_pe = aEnergy;
    // we want to use NDC coordinates so do perspective division manually
    window_pos_x = int(window_width_half * gl_Position.x / gl_Position.w + window_width_half);
    window_pos_y = int(window_height_half * gl_Position.y / gl_Position.w + window_height_half);

}