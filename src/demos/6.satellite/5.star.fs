#version 330 core
out vec4 FragColor;
flat in int window_pos_x;
flat in int window_pos_y;
in float n_pe;
flat in int ptSize;
flat in int highlightType;
uniform float n_e;  // photons per grayscale
layout (std140) uniform energyProportion    //  stor energy proportion of each pixel, starts from lower-left most pixel
{
    //!float ep[25]; // std140 alignment float is same as vec4, so if we choose float, we will lose 3 component values
    vec4 ep[36]; // 12*12/4=36, [36] is big enough to store 12*12 pixels information
    // e.g 5*5
    // pixel(0.5,0.5) -> ep[0].x
    // pixel(0.5,1.5) -> ep[0].y
    // pixel(0.5,2.5) -> ep[0].z
    // pixel(0.5,3.5) -> ep[0].w
    // pixel(0.5,4.5) -> ep[1].x
    // pixel(1.5,0.5) -> ep[1].y
};

void main()
{   
    FragColor = vec4(1.0,0.0,0.0,1.0);
    // the (x, y) location (0.5, 0.5) is returned for the lower-left-most pixel in a window
    // vec2 window_pos_offset;
    // window_pos_offset.x = gl_FragCoord.x - window_pos.x;
    // window_pos_offset.y = gl_FragCoord.y - window_pos.y;

    int curPixel_x = int(gl_FragCoord.x);
    int curPixel_y = int(gl_FragCoord.y);
    int window_pos_offset_x = curPixel_x - window_pos_x;
    int window_pos_offset_y = curPixel_y - window_pos_y;

    // The int function works just like floor; we add 0.5 to deal with the rounding-up case.
    // int offset_index_x = int(abs(window_pos_offset.x));

    // int offset_index_y = int(abs(window_pos_offset.y));

    // if (window_pos_offset.x < 0)
    // {
    //     offset_index_x = -offset_index_x;
    // }
    // if (window_pos_offset.y < 0)
    // {
    //     offset_index_y = -offset_index_y;
    // }

    // int matrix_index_x = int(ptSize)/2 + offset_index_x;
    // int matrix_index_y = int(ptSize)/2 + offset_index_y;

    int matrix_index_x = int(ptSize)/2 + window_pos_offset_x;
    int matrix_index_y = int(ptSize)/2 + window_pos_offset_y;

    int epIndex = matrix_index_x * ptSize + matrix_index_y;

    int epIndex_row = epIndex / 4;
    int enIndex_col = epIndex % 4;

    float photons = n_pe * ep[epIndex_row][enIndex_col];

    float grayscale = photons / n_e / 256.0;
    //grayscale = ep[epIndex_row][enIndex_col];
    // try to fix overlay artifact
    if (grayscale < 0.001)
    {
        FragColor = vec4(grayscale, grayscale, grayscale, 0.0);
    }
    else
    {
        FragColor = vec4(grayscale, grayscale, grayscale, 1.0);
    }
    
    if (highlightType == 1)
    {
        FragColor = vec4(grayscale, 0.0 ,0.0 ,1.0);
    }
    else
    {
        //FragColor = vec4(1.0, 0.0 ,0.0 ,1.0);
    }
}