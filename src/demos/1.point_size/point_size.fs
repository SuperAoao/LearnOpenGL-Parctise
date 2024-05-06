#version 330 core
out vec4 FragColor;
flat in int isGreen;

void main()
{   
    if (isGreen == 1)
    {
         FragColor = vec4(0.0,1.0,0.0,1.0);
    }
    else
    {
         FragColor = vec4(1.0,0.0,0.0,1.0);
    }
}