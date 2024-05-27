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
flat out int highlightType;
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

    // highlight vega
    highlightType = 0;
    vec3 vega = vec3(3.16450477, 22.2810555, 11.5554714);
    vec3 dis = abs(vega-aPos);
    float nVega = 66126308.230977856;
    float energyOffset = abs(nVega - aEnergy);
    if (dis.x < 0.1 && dis.y < 0.1 && dis.z < 0.1 && energyOffset < 0.1)
    {
        highlightType = 1;
    }
    vec3 kLyra = vec3(16.6457272, 204.706192, 120.070076);
    dis = abs(kLyra - aPos);
    if (dis.x < 0.1 && dis.y < 0.1 && dis.z < 0.1 )
    {
        highlightType = 1;
    }

    vec3 epsilonLyra = vec3(24.0562897, 143.722794, 71.3953781);
    dis = abs(epsilonLyra - aPos);
    if (dis.x < 0.1 && dis.y < 0.1 && dis.z < 0.1 )
    {
        highlightType = 1;
    }

    vec3 Zeta1 = vec3(23.6462040, 133.493820, 72.2769241);
    dis = abs(Zeta1 - aPos);
    if (dis.x < 0.1 && dis.y < 0.1 && dis.z < 0.1 )
    {
        highlightType = 1;
    }

    
    vec3 Delta2 = vec3(169.281799, 772.742249, 426.113007);
    dis = abs(Delta2 - aPos);
    if (dis.x < 0.1 && dis.y < 0.1 && dis.z < 0.1 )
    {
        highlightType = 1;
    }
    
    vec3 Sulafat = vec3(135.845078, 519.888489, 337.544525);
    dis = abs(Sulafat - aPos);
    if (dis.x < 0.1 && dis.y < 0.1 && dis.z < 0.1 )
    {
        highlightType = 1;
    }

    vec3 Sheliak = vec3(159.606888, 730.687195, 466.606110);
    dis = abs(Sheliak - aPos);
    if (dis.x < 0.1 && dis.y < 0.1 && dis.z < 0.1 )
    {
        highlightType = 1;
    }
}