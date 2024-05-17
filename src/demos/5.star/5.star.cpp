/*
* This is a star simulator demo according to "High-Fidelity Star Simulator for Cameras and Star Trackers" 
*/
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <stb_image.h>
#include <stb_image_write.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <learnopengl/filesystem.h>
#include <learnopengl/shader.h>
#include <learnopengl/camera.h>
#include <learnopengl/model.h>

#include <iomanip>
#include <iostream>
#include <random>

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow *window);
unsigned int loadTexture(const char *path);

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

// camera
Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));
float lastX = (float)SCR_WIDTH  / 2.0;
float lastY = (float)SCR_HEIGHT / 2.0;
bool firstMouse = true;

// timing
float deltaTime = 0.0f;
float lastFrame = 0.0f;

// gaussian parameters
const double g_sigma = 1.12;
const double g_sigma_squared = g_sigma * g_sigma;
const uint32_t g_pixleSize_min = 3;
const uint32_t g_pixleSize_max = 9;

// integration steps size
const uint32_t g_steps = 100;

// exposure time 1s, d_length 1 mm^2
const double g_CCDphotons = 19100;
// get photons from star magitude on CCD
double getPhotonsFromMagnitude(double mv, double d_len, double t_in)
{
    double n_pe = g_CCDphotons * 1 / std::pow(2.5, mv) * t_in * glm::pi<double>() * (d_len / 2) * (d_len / 2) * 1e6;
    return n_pe;
}
// we consider center point (xc,yc) = (0,0)
double simplifiedGaussianFunc(double x, double y, double xc, double yc)
{
    double xDis = x - xc;
    double yDis = y - yc;
    return std::exp(-(xDis * xDis + yDis * yDis) / (2 * g_sigma_squared));
}

uint32_t getPointSizeFromSigma(double sigma)
{
    // from "High-Fidelity Star Simulator for Cameras and Star Trackers" p49
    uint32_t pixelSize = std::floor(sigma * 4.48 + 0.5);   // +0.5 consider rounding up
    pixelSize = std::clamp(pixelSize, g_pixleSize_min, g_pixleSize_max);
    return pixelSize;
}

double energyProportionInPixel(double x, double y)
{
    // gaussian center is (0,0), and our center pixel block is (0, 0)
    // range is 1, so we start from x-0.5,y-0.5
    x -= 0.5;
    y -= 0.5;
    double hx = 1 / (double)g_steps; // x 方向步长
    double hy = 1 / (double)g_steps; // y 方向步长
    double xc = 0.0;
    double yc = 0.0;
    double sum = 0.0;
    for (uint32_t i = 0; i < g_steps; ++i)
    {
        for (uint32_t j = 0; j < g_steps; ++j)
        {
            double x0 = x + i * hx; // 当前子区间左边界的 x 值
            double x1 = x + (i + 1) * hx; // 当前子区间右边界的 x 值
            double y0 = y + j * hy; // 当前子区间下边界的 y 值
            double y1 = y + (j + 1) * hy; // 当前子区间上边界的 y 值

            sum += (simplifiedGaussianFunc(x0, y0, xc, yc) + 
                simplifiedGaussianFunc(x1, y0, xc, yc) +
                simplifiedGaussianFunc(x0, y1, xc, yc) +
                simplifiedGaussianFunc(x1, y1, xc, yc)) / 4.0 * hx * hy;
        }
    }
    sum = 1 / (2 * glm::pi<double>() * g_sigma_squared) * sum;
    return sum;
}

// post process for noise 
float frameVertics[] = {
    // pos                // texCoords
    -1.0f, -1.0f, 0.0f,     0.0f, 0.0f,
    1.0f, -1.0f, 0.0f,      1.0f, 0.0f,
    -1.0f, 1.0f, 0.0f,      0.0f, 1.0f,
    -1.0f, 1.0f, 0.0f,      0.0f, 1.0f,
    1.0f, -1.0f, 0.0f,      1.0f, 0.0f,
    1.0f, 1.0f, 0.0f,       1.0f, 1.0f
};

std::vector<float> genEnergyProportion(uint32_t pixelSize, double n_pe, double n_e)
{
    std::vector<float> vecEnergyProportion;
    std::vector<float> vecGrayScale;
    double sum = 0.0;
    const int center_x = pixelSize/2;
    const int center_y = pixelSize/2;
    for (uint32_t i = 0; i < pixelSize; ++i)
    {
        for (uint32_t j = 0; j < pixelSize; ++j)
        {
            double energyProportion = energyProportionInPixel((double)i - center_x, (double)j - center_x);
            sum += energyProportion;
            float nPhotons = n_pe * energyProportion; // 该像素的光子数
            float nGrayScale = nPhotons / n_e;
            vecEnergyProportion.push_back((float)energyProportion);
            vecGrayScale.push_back(nGrayScale);
            //-----------------------------------------------------------------------------
            std::cout << "The pixel block(" << i << "," << j << "): " << std::endl;
            std::cout << "  energy proportion: " << energyProportion << std::endl;
            std::cout << "  phontons: " << nPhotons << std::endl;
            std::cout << "  gray value: " << nGrayScale << std::endl;
            //-----------------------------------------------------------------------------
        }
    }

    std::cout << "energy sum: " << sum << std::endl;
    // print grayscale
    std::cout << "--------print graylevel of each pixel--------" << std::endl;
    for (int i = pixelSize-1; i >=0; --i)
    {
        for (int j = 0; j < pixelSize; ++j)
        {
            std::cout << std::setw(12) << vecGrayScale[i * pixelSize + j] << "   ";
        }
        std::cout << std::endl;
    }
    return vecEnergyProportion;
}

int main()
{
    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // glfw window creation
    // --------------------
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "star", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // tell GLFW to capture our mouse
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // configure global opengl state
    // -----------------------------
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_ALWAYS); // always pass the depth test (same effect as glDisable(GL_DEPTH_TEST))

    // build and compile shaders
    // -------------------------
    Shader shaderStar("5.star.vs", "5.star.fs");
    Shader shaderNoise("5.noise.vs", "5.noise.fs");
    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    float pointVertices[] = {
        // positions        
        -0.5f, -0.5f, -0.5f,
         0.5f, 0.5f, -0.5f,
    };

    double n_ec = 45000;    // CCD per pixel max photons
    double n_e = n_ec / 256;    // photons per gray scale
    double d_len = 0.07; // unit m
    double t_exposure = 5;    // unit s
    double m_v = 12;    //  star magnitude
    double n_pe = getPhotonsFromMagnitude(m_v, d_len, t_exposure);
    double test = 1 / (2 * glm::pi<double>() * g_sigma_squared) * simplifiedGaussianFunc(0, 0, 0, 0) * n_pe;
    // test gaussian

    uint32_t pointSize = getPointSizeFromSigma(g_sigma);
    //test 
    pointSize = 12;
    std::vector<float> vecVertics;
    vecVertics.push_back(pointVertices[0]);
    vecVertics.push_back(pointVertices[1]);
    vecVertics.push_back(pointVertices[2]);
    vecVertics.push_back((float)n_pe);

    std::vector<float> vecEnergyProportion = genEnergyProportion(pointSize, n_pe, n_e);

    // noise
    double n_pe_background = getPhotonsFromMagnitude(10, d_len, t_exposure); // 背景电子数 
    double n_background = n_pe_background * 0.13;   //平均背景电子数
    //double S_AvgShot = n_pe / (SCR_WIDTH * SCR_HEIGHT) + n_background; 
    double S_AvgShot = n_pe + n_pe_background;
    double sigma_PRNU = 0.03;
    double I_dark = 203;
    double I_DSNU = 60;
    double N_well = 70000;
    double q = 12;
    double I_R = 2;
    double n_shot = sqrt(S_AvgShot); // 散粒噪声
    double n_shot_background = sqrt(n_pe_background);
    double n_PRNU = sqrt(S_AvgShot * sigma_PRNU); // 光响应非均匀噪声
    double n_DS = sqrt(I_dark * t_exposure); // 暗电流噪声
    double n_DSNU = sqrt(I_DSNU * t_exposure); // 暗电流不均匀噪声
    double n_ADC = N_well / pow(2,q) / sqrt(12); // 模拟数字转化噪声
    double n_R = I_R; // 读出噪声
    double n_TOT = sqrt(pow(n_shot_background,2) + pow(n_PRNU,2) + pow(n_DS,2) + pow(n_DSNU, 2) + pow(n_ADC, 2) + pow(n_R, 2)); // 噪声标准差

    std::random_device rd{};
    std::mt19937 gen{ rd() };

    // values near the mean are the most likely
    // standard deviation affects the dispersion of generated values from the mean
    std::normal_distribution d{ n_pe, n_shot };
    float shot_n_pe = std::round(d(gen));
    std::normal_distribution noise{ 0.0, n_TOT };
    // gen noise texture

    float n_perPixel = std::round(noise(gen));
    //---------------------------------------
    // star point VAO
    unsigned int pointVAO, pointVBO;
    glGenVertexArrays(1, &pointVAO);
    glGenBuffers(1, &pointVBO);
    glBindVertexArray(pointVAO);
    glBindBuffer(GL_ARRAY_BUFFER, pointVBO);

    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vecVertics.size(), vecVertics.data(), GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(3 * sizeof(float)));

    glBindVertexArray(0);
    
    // noise VAO
    unsigned int noiseVAO, noiseVBO;
    glGenVertexArrays(1, &noiseVAO);
    glGenBuffers(1, &noiseVBO);
    glBindVertexArray(noiseVAO);
    glBindBuffer(GL_ARRAY_BUFFER, noiseVBO);

    glBufferData(GL_ARRAY_BUFFER, sizeof(frameVertics), &frameVertics, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3*sizeof(float)));
    glBindVertexArray(0);

    // configure a unifom buffer object to store energy proportion of each pixel
    // ------------------------------
    unsigned int uniformBlockIndex = glGetUniformBlockIndex(shaderStar.ID, "energyProportion");
    glUniformBlockBinding(shaderStar.ID, uniformBlockIndex, 0);
    unsigned int uboEnergyProportion;
    glGenBuffers(1, &uboEnergyProportion);
    glBindBuffer(GL_UNIFORM_BUFFER, uboEnergyProportion);
    glBufferData(GL_UNIFORM_BUFFER, sizeof(float) * vecEnergyProportion.size(), vecEnergyProportion.data(), GL_DYNAMIC_DRAW);
    glBindBufferRange(GL_UNIFORM_BUFFER, 0, uboEnergyProportion, 0, sizeof(float) * vecEnergyProportion.size());
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    glBindVertexArray(0);

    // generate fbo for first render pass of star image
    unsigned int fboStar;
    glGenFramebuffers(1, &fboStar);
    glBindFramebuffer(GL_FRAMEBUFFER, fboStar);

    unsigned int texColorBuffer;
    glGenTextures(1, &texColorBuffer);
    glBindTexture(GL_TEXTURE_2D, texColorBuffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texColorBuffer, 0);


    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // enable gl_PointSize
    glEnable(GL_PROGRAM_POINT_SIZE);
    // render loop
    // -----------
    while(!glfwWindowShouldClose(window))
    {
        // per-frame time logic
        // --------------------
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // input
        // -----
        processInput(window);

        // render
        // ------
        glBindFramebuffer(GL_FRAMEBUFFER, fboStar);
        shaderStar.use();
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        shaderStar.use();
        glm::mat4 model = glm::mat4(1.0f);
        glm::mat4 view = camera.GetViewMatrix();
        glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
 
        shaderStar.setMat4("view", view);
        shaderStar.setMat4("projection", projection);
        shaderStar.setFloat("window_width_half", (float)SCR_WIDTH / 2);
        shaderStar.setFloat("window_height_half", (float)SCR_HEIGHT / 2);
        shaderStar.setFloat("n_e", (float)n_e);
        shaderStar.setInt("pointSize", pointSize);

        glBindVertexArray(pointVAO);

        model = glm::translate(model, glm::vec3(0.0f, 0.0f, 0.0f));
        shaderStar.setMat4("model", model);
        glDrawArrays(GL_POINTS, 0, 1);

        glBindVertexArray(0);

        // debug 
        glm::vec4 pos(pointVertices[0], pointVertices[1], pointVertices[2], 1.0);
       
        glm::vec4 gl_Pos = projection * view * model * pos;
        float gl_window_x = (float)SCR_WIDTH / 2 * gl_Pos.x / gl_Pos.w + (float)SCR_WIDTH / 2;
        float gl_window_y = (float)SCR_HEIGHT / 2 * gl_Pos.y / gl_Pos.w + (float)SCR_HEIGHT / 2;
        std::cout << "Point 0: " << std::endl;
        //  OpenGL then performs perspective division on the clip-space coordinates to transform them to normalized-device coordinates
        std::cout << "  gl_Pos: " << gl_Pos.x << ", " << gl_Pos.y << ", " << gl_Pos.z <<  ", " << gl_Pos.w << std::endl;
        std::cout << "  window_Pos: " << gl_window_x << ", " << gl_window_y << std::endl;

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glBindVertexArray(noiseVAO);
        shaderNoise.use();
        glUniform1i(glGetUniformLocation(shaderNoise.ID, "screenTexture"), 0);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glBindVertexArray(0);
        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // optional: de-allocate all resources once they've outlived their purpose:
    // ------------------------------------------------------------------------
    glDeleteVertexArrays(1, &pointVAO);
    glDeleteBuffers(1, &pointVBO);

    glfwTerminate();
    return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.ProcessKeyboard(FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.ProcessKeyboard(BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.ProcessKeyboard(LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.ProcessKeyboard(RIGHT, deltaTime);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback(GLFWwindow* window, double xposIn, double yposIn)
{
    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);

    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

    lastX = xpos;
    lastY = ypos;

    camera.ProcessMouseMovement(xoffset, yoffset);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    camera.ProcessMouseScroll(static_cast<float>(yoffset));
}

// utility function for loading a 2D texture from file
// ---------------------------------------------------
unsigned int loadTexture(char const *path)
{
    unsigned int textureID;
    glGenTextures(1, &textureID);

    int width, height, nrComponents;
    unsigned char *data = stbi_load(path, &width, &height, &nrComponents, 0);
    if (data)
    {
        GLenum format;
        if (nrComponents == 1)
            format = GL_RED;
        else if (nrComponents == 3)
            format = GL_RGB;
        else if (nrComponents == 4)
            format = GL_RGBA;

        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        stbi_image_free(data);
    }
    else
    {
        std::cout << "Texture failed to load at path: " << path << std::endl;
        stbi_image_free(data);
    }

    return textureID;
}
