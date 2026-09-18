#include "app/render_loop.h"

#include "app/gl_resources.h"
#include "app/render_imgui.h"
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
#include "ImGui/imgui.h"
#include "ImGui/imgui_impl_glfw.h"
#include "ImGui/imgui_impl_opengl3.h"

void mainLoop()
{
    AppState& app = AppState::Get();
    while (!glfwWindowShouldClose(app.window))
    {
        glfwPollEvents();

        runCuda();

        std::string title = "Luxion | " + utilityCore::convertIntToString(app.iteration) + " Iterations";
        glfwSetWindowTitle(app.window, title.c_str());
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, app.pbo);
        glBindTexture(GL_TEXTURE_2D, app.displayImage);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, app.width, app.height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glClear(GL_COLOR_BUFFER_BIT);

        // Binding GL_PIXEL_UNPACK_BUFFER back to default
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        // VAO, shader program, and texture already bound
        glDrawElements(GL_TRIANGLES, 6,  GL_UNSIGNED_SHORT, 0);

        // Render ImGui Stuff
        RenderImGui();

        glfwSwapBuffers(app.window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(app.window);
    glfwTerminate();
}


void runCuda()
{
    AppState& app = AppState::Get();
    if (app.camchanged)
    {
        app.iteration = 0;
        Camera& cam = app.renderState->camera;

        glm::vec3 focus_point = cam.lookAt;
        glm::vec3 sphericals;

        cam.view = cam.lookAt - cam.position;

        sphericals.x = -app.zoom * sin(app.phi) * sin(app.theta);
        sphericals.y = -app.zoom * cos(app.theta);
        sphericals.z = app.zoom * cos(app.phi) * sin(app.theta);

        // fmt::println("OG: {}", glm::to_string(cam.view));
        // fmt::println("NEW: {}", glm::to_string(-glm::normalize(sphericals)));

        cam.view = -glm::normalize(sphericals);
        glm::vec3 v = cam.view;
        glm::vec3 u = app.refUp - v * glm::dot(app.refUp, v);
        if (glm::pow(glm::length(u), 2.0f) < 1e-6f) {
            u = glm::vec3(0, 1, 0);
        }
        u = glm::normalize(u);
        glm::vec3 r = glm::normalize(glm::cross(v, u));
        u = glm::cross(r, v);
        cam.up    = u;
        cam.right = r;

        cam.position = cam.lookAt - cam.view * app.zoom;
        app.camchanged = false;
    }

    // Map OpenGL buffer object for writing from CUDA on a single GPU
    // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer

    if (app.iteration == 0)
    {
        pathtraceFree();
        pathtraceInit(app.scene);
    }

    if (app.iteration < app.renderState->iterations)
    {
        uchar4* pbo_dptr = NULL;
        app.iteration++;
        cudaGLMapBufferObject((void**)&pbo_dptr, app.pbo);

        // execute the kernel
        int frame = 0;
        pathtrace(pbo_dptr, frame, app.iteration);

        // unmap buffer object
        cudaGLUnmapBufferObject(app.pbo);
    }
    else
    {
        saveImage();
        pathtraceFree();
        cudaDeviceReset();
        exit(EXIT_SUCCESS);
    }
}
