#include "app/save_image.h"

#include "app/app_state.h"

#include "glslUtility.hpp"
#include "image.h"
#include "pathtrace.h"
#include "scene.h"
#include "sceneStructs.h"
#include "utilities.h"
#include "myoptix.h"
#include "config.h"

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

glm::vec3 ACESFilmHost(glm::vec3 x) {
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return glm::clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0f, 1.0f);
}


void saveImage()
{
    AppState& app = AppState::Get();
    copyImageToHost();

    float samples = static_cast<float>(app.iteration);
    // output image file
    Image img(app.width, app.height);

    for (int x = 0; x < app.width; x++)
    {
        for (int y = 0; y < app.height; y++)
        {
            int index = x + (y * app.width);
            glm::vec3 pix = app.renderState->image[index] / samples;

            //reinhard op
            // pix = pix / (pix + glm::vec3(1.0f));
            pix = ACESFilmHost(pix);

            //gamma correction
            pix = glm::pow(pix, glm::vec3(0.45f));

            img.setPixel(app.width - 1 - x, y, pix);
        }
    }

    std::string filename;
    if (!app.outputName.empty())
    {
        filename = app.outputName;
    }
    else
    {
        std::ostringstream ss;
        ss << "img/" << app.renderState->imageName << "." << app.startTimeString << "." << samples << "samp";
        filename = ss.str();
    }

    // CHECKITOUT
    img.savePNG(filename);
    //img.saveHDR(filename);  // Save a Radiance HDR file
}
