/*
* This is a star simulator demo according to "High-Fidelity Star Simulator for Cameras and Star Trackers" 
*/
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
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

#include "5.stardb.h"

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow *window);
unsigned int loadTexture(const char *path);

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 800;

// camera
glm::vec3 vegaPos(3.16450477, 22.2810555, 11.5554714);
glm::vec3 vecFront(1.0, 0.0, 0.0);
glm::vec3 vecUp(0.0, 1.0, 0.0);
// test yaw=-90.0, pitch = 0.0
glm::vec3 vecTestFront(0.0, 0.0, -1.0);
glm::vec3 vecTestRight(1.0, 0.0, 0.0);

glm::vec3 vega_xz(3.16450477, 0.0, 11.5554714); // ignore y component

//float cosYaw = glm::dot(vecFront, vecTestFront) / (glm::length(vecTestFront) * glm::length(vecFront));
//float yaw = glm::acos(cosYaw);
//float yaw_deg = glm::degrees(yaw);    // remember to consider about sign
//
//float cosPitch = glm::dot(vecTestFront, vecTestFront) / (glm::length(vecTestFront) * glm::length(vecTestFront));
//float pitch = glm::acos(cosPitch);
//float pitch_deg = glm::degrees(pitch);

float cosYaw = glm::dot(vega_xz, vecFront) / (glm::length(vega_xz) * glm::length(vecFront) );
float yaw = glm::acos(cosYaw);
float yaw_deg = glm::degrees(yaw);

float cosPitch = glm::dot(vegaPos, vega_xz) / (glm::length(vegaPos) * glm::length(vega_xz));
float pitch = glm::acos(cosPitch);
float pitch_deg = glm::degrees(pitch);

Camera camera(glm::vec3(0.0f, 0.0f, 0.0f), vecUp, yaw_deg, pitch_deg);
float lastX = (float)SCR_WIDTH  / 2.0;
float lastY = (float)SCR_HEIGHT / 2.0;
bool firstMouse = true;

// timing
float deltaTime = 0.0f;
float lastFrame = 0.0f;

// black theory
const double g_h = 299792458.0; // Planck constant measured in[J ·s]
const double g_c = 6.62607015e-34; //the speed of light constant measure in[m/s]
const double g_k = 1.380649e-23; //Boltzmann constant measured in [J/K];

// satellite
const double g_E_sun = 573; // Irradiance of the sun measured in[W/m^2]
const double g_sun_app_mag = -26.7; // sun apparent magnitude

// CCD gray parameters
const double n_ec = 45000;    // CCD per pixel max photons
const double n_e = n_ec / 256;    // photons per gray scale

// gaussian parameters
const double g_sigma = 0.85;
const double g_sigma_squared = g_sigma * g_sigma;
const uint32_t g_pixleSize_min = 3;
const uint32_t g_pixleSize_max = 9;

// integration steps size
const uint32_t g_steps = 100;

// exposure time 1s, d_length 1 mm^2
const double g_CCDphotons = 19100;

// in [m]
double g_viewDistance_r = 20440.0;

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

bool createNoiseTex(const double d_len, const double t_exposure)
{
    bool ret = false;

    size_t imgSize = SCR_WIDTH * SCR_HEIGHT;
    unsigned char* data = new unsigned char[imgSize];
    
    if (!data)
    {
        std::cout << "[Error]can not allocate noise texure" << std::endl;
        return ret;
    }
    // noise
    double n_pe_background = getPhotonsFromMagnitude(12, d_len, t_exposure); // 背景电子数 
    n_pe_background = 3000.0;
    //double n_background = n_pe_background * 0.13;   //平均背景电子数
    //double S_AvgShot = n_pe / (SCR_WIDTH * SCR_HEIGHT) + n_background; 
    double S_AvgShot = n_pe_background;
    double sigma_PRNU = 0.03;
    double I_dark = 203;
    double I_DSNU = 60;
    double N_well = 10000;
    double q = 12;
    double I_R = 2;

    double n_shot_background = sqrt(n_pe_background);// 散粒背景噪声
    double n_PRNU = sqrt(S_AvgShot * sigma_PRNU); // 光响应非均匀噪声
    double n_DS = sqrt(I_dark * t_exposure); // 暗电流噪声
    double n_DSNU = sqrt(I_DSNU * t_exposure); // 暗电流不均匀噪声
    double n_ADC = N_well / pow(2, q) / sqrt(12); // 模拟数字转化噪声
    double n_R = I_R; // 读出噪声
    double n_TOT = sqrt(pow(n_shot_background, 2) + pow(n_PRNU, 2) + pow(n_DS, 2) + pow(n_DSNU, 2) + pow(n_ADC, 2) + pow(n_R, 2)); // 噪声标准差

    std::random_device rd{};
    std::mt19937 gen{ rd() };

    // values near the mean are the most likely
    // standard deviation affects the dispersion of generated values from the mean

    std::normal_distribution noise{ n_pe_background, n_TOT };
    
    for (unsigned int i = 0; i < imgSize; ++i)
    {
        //float n_noise = std::max(std::round(noise(gen)), 0.0);
        float n_noise = std::round(noise(gen));
        data[i] = n_noise / n_e;
    }

    stbi_write_png("noiseTex.png", SCR_WIDTH, SCR_HEIGHT, 1, data, sizeof(unsigned char) * SCR_WIDTH);
    delete []data;
    return ret;
}

// p28
double computeGAGS(const double d_len, const double t_exposure, double gain, double lambda_avg, double QE_avg, double tau_opt, double F_m)
{
    double g = std::_Pi_val * 0.25 * d_len * d_len * t_exposure / gain * F_m * lambda_avg / (g_h * g_c) * QE_avg * tau_opt;
    return g;
}

/**
 * @brief 
 * @param r -> the distance between the target and the sensor [m]
 * @param rho -> effective reflectance 
 * @param area -> effective reflective area [m^2]
 * @return 
 */
double computeIrradianceOfSatellite(const double r,const double rho, const double area)
{
    double E = g_E_sun / (std::_Pi_val * r * r) * rho * area;
    return E;
}

double irradianceToApparentMagnitude(const double e)
{
    double m = 2.5 * std::log10(g_E_sun / e) + g_sun_app_mag;
    return m;
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


    double d_len = 0.03; // unit m
    double t_exposure = 5;    // unit s
    double m_v = 0.03;    //  star magnitude
    double n_pe = getPhotonsFromMagnitude(12.0, d_len, t_exposure);
    double test = 1 / (2 * glm::pi<double>() * g_sigma_squared) * simplifiedGaussianFunc(0, 0, 0, 0) * n_pe;
    // test gaussian

    uint32_t pointSize = getPointSizeFromSigma(g_sigma);
    pointSize = 12;
    //pointSize = 12; // from the visual data from the second output is 5
    //test 
    std::vector<float> vecVertics;
    vecVertics.push_back(pointVertices[0]);
    vecVertics.push_back(pointVertices[1]);
    vecVertics.push_back(pointVertices[2]);
    vecVertics.push_back((float)n_pe);

    std::vector<float> vecEnergyProportion = genEnergyProportion(pointSize, n_pe, n_e);


    // gen noise texture
    //double n_shot = sqrt(S_AvgShot); // 散粒噪声
    //std::normal_distribution d{ n_pe, n_shot };
    //     float shot_n_pe = std::round(d(gen));

    createNoiseTex(d_len, t_exposure);

    StarDataBase db;
    db.loadBinaryData("stars.dat");
    std::vector<float> vecStarsVertics = db.getVerticsArray(d_len, t_exposure);
    // 
    //---------------------------------------
    // star point VAO
    unsigned int pointVAO, pointVBO;
    glGenVertexArrays(1, &pointVAO);
    glGenBuffers(1, &pointVBO);
    glBindVertexArray(pointVAO);
    glBindBuffer(GL_ARRAY_BUFFER, pointVBO);

    //glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vecVertics.size(), vecVertics.data(), GL_DYNAMIC_DRAW);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vecStarsVertics.size(), vecStarsVertics.data(), GL_DYNAMIC_DRAW);
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

    // satellite
    // compute the satellite effective reflective area
    double refArea = 1.0 * 1.0 + 5.0 * 1.0 * 2.0; // 1 plane + 2 wings 
    // calculate irradiance of satellite 
    //double E = computeIrradianceOfSatellite(648610.0, 0.25, 0.290);
    double E = computeIrradianceOfSatellite(g_viewDistance_r, 0.25, 0.800);
    double target_mag = irradianceToApparentMagnitude(E);

    unsigned int satelliteVAO, satelliteVBO;
    glGenVertexArrays(1, &satelliteVAO);
    glGenBuffers(1, &satelliteVBO);
    glBindVertexArray(satelliteVAO);
    glBindBuffer(GL_ARRAY_BUFFER, satelliteVBO);
    std::vector<float> vecSatelliteVertics;
    vecSatelliteVertics.push_back(3.16450477);
    vecSatelliteVertics.push_back(22.2810555);
    vecSatelliteVertics.push_back(11.5554714);
    vecSatelliteVertics.push_back(getPhotonsFromMagnitude(target_mag,d_len, t_exposure));
    glBufferData(GL_ARRAY_BUFFER, sizeof(float)* vecSatelliteVertics.size(), vecSatelliteVertics.data(), GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(3 * sizeof(float)));
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

    stbi_set_flip_vertically_on_load(true); // tell stb_image.h to flip loaded texture's on the y-axis.
    unsigned int texNoise = loadTexture("noiseTex.png");
    
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // enable gl_PointSize
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_DEPTH_TEST);

   

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
        glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 1500.0f);
 
        shaderStar.setMat4("view", view);
        shaderStar.setMat4("projection", projection);
        shaderStar.setFloat("window_width_half", (float)SCR_WIDTH / 2);
        shaderStar.setFloat("window_height_half", (float)SCR_HEIGHT / 2);
        shaderStar.setFloat("n_e", (float)n_e);
        shaderStar.setInt("pointSize", pointSize);

        glBindVertexArray(pointVAO);

        model = glm::translate(model, glm::vec3(0.0f, 0.0f, 0.0f));
        shaderStar.setMat4("model", model);
        //glDrawArrays(GL_POINTS, 0, db.getStarNum());
        // 
        // draw satellite
        glBindVertexArray(satelliteVAO);
        // update
        double E = computeIrradianceOfSatellite(g_viewDistance_r, 0.25, 0.800);
        double target_mag = irradianceToApparentMagnitude(E);

        glBindBuffer(GL_ARRAY_BUFFER, satelliteVBO);
 
        std::vector<float> vecSatelliteVertics;
        vecSatelliteVertics.push_back(3.16450477);
        vecSatelliteVertics.push_back(22.2810555);
        vecSatelliteVertics.push_back(11.5554714);
        vecSatelliteVertics.push_back(getPhotonsFromMagnitude(target_mag, d_len, t_exposure));
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float) * vecSatelliteVertics.size(),vecSatelliteVertics.data());

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
        // bind textures on corresponding texture units
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texColorBuffer);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, texNoise);
        glUniform1i(glGetUniformLocation(shaderNoise.ID, "screenTexture"), 0);
        glUniform1i(glGetUniformLocation(shaderNoise.ID, "noiseTexture"), 1);
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
    g_viewDistance_r += yoffset * (-10000.0f);
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
