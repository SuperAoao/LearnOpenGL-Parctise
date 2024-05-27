#version 330 core
out vec4 FragColor;
in vec2 texCoords;

uniform sampler2D screenTexture;
uniform sampler2D noiseTexture;
void main()
{   
    FragColor = texture(screenTexture, texCoords);
    vec4 noise = texture(noiseTexture, texCoords);
    FragColor.r += noise.r;
    FragColor.g += noise.r;
    FragColor.b += noise.r;
    //FragColor = vec4(0.001,0.001,0.001,1.0);
}