#version 330 core
layout (location = 0) in vec3 aPos;


uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
// Won't be interpolated in rasterization
flat out int isGreen;

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    if (gl_Position.x > 0.0)
    {
        gl_PointSize = 100;
    }
    else
    {
        gl_PointSize = 10;
    }
    
    if (gl_VertexID > 0)
    {
        isGreen = 1;
    }
    else
    {
        isGreen = 0;
    }

}