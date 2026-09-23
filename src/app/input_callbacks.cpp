#include "app/input_callbacks.h"

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
#include "ImGui/imgui.h"

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

//-------------------------------
//------INTERACTIVITY SETUP------
//-------------------------------

bool shiftPressed(GLFWwindow* window)
{
    return glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
           glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    AppState& app = AppState::Get();
    if (action == GLFW_PRESS)
    {
        // Camera& cam = renderState->camera;
        switch (key)
        {
            case GLFW_KEY_ESCAPE:
                saveImage();
                glfwSetWindowShouldClose(window, GL_TRUE);
                break;
            case GLFW_KEY_I:
                saveImage();
                break;
            case GLFW_KEY_H:
                if (!app.io->WantCaptureKeyboard)
                {
                    app.hideImGui = !app.hideImGui;
                }
                break;
            case GLFW_KEY_SPACE:
                if (app.locked)
                {
                    break;
                }
                app.camchanged = true;
                app.renderState = &app.scene->state;
                break;
        }
    }
}

void processKeyboardMovement(GLFWwindow* window, float dt)
{
    AppState& app = AppState::Get();
    if (app.locked || app.io->WantCaptureKeyboard)
    {
        return;
    }

    auto keyDown = [window](int key) { return glfwGetKey(window, key) == GLFW_PRESS; };

    Camera& cam = app.renderState->camera;

    glm::vec3 move(0.0f);
    if (keyDown(GLFW_KEY_W)) {
        move += cam.view;
    }
    if (keyDown(GLFW_KEY_S)) {
        move -= cam.view;
    }
    if (keyDown(GLFW_KEY_D)) {
        move += cam.right;
    }
    if (keyDown(GLFW_KEY_A)) {
        move -= cam.right;
    }
    if (keyDown(GLFW_KEY_E)) {
        move += app.refUp;
    }
    if (keyDown(GLFW_KEY_Q)) {
        move -= app.refUp;
    }

    if (move != glm::vec3(0.0f))
    {
        const float moveSpeed = 1.0f;
        cam.lookAt += move * (app.zoom * moveSpeed * dt);
        app.camchanged = true;
    }

    const float orbitSpeed = 1.5f;
    float dphi = 0.0f, dtheta = 0.0f;
    if (keyDown(GLFW_KEY_LEFT))  { dphi   -= orbitSpeed * dt; }
    if (keyDown(GLFW_KEY_RIGHT)) { dphi   += orbitSpeed * dt; }
    if (keyDown(GLFW_KEY_UP))    { dtheta -= orbitSpeed * dt; }
    if (keyDown(GLFW_KEY_DOWN))  { dtheta += orbitSpeed * dt; }

    if (dphi != 0.0f || dtheta != 0.0f)
    {
        app.phi += dphi;
        app.theta = glm::clamp(app.theta + dtheta, 0.001f, PI - 0.001f);
        app.camchanged = true;
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    AppState& app = AppState::Get();
    if (MouseOverImGuiWindow())
    {
        return;
    }

    app.leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
    app.rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
    app.middleMousePressed = (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos)
{
    AppState& app = AppState::Get();
    if (xpos == app.lastX && ypos == app.lastY)
        return;

    double dx = xpos - app.lastX;
    double dy = ypos - app.lastY;

    if (MouseOverImGuiWindow() || app.locked)
        goto end;

    // SHIFT + LEFT DRAG -> PAN
    if (app.leftMousePressed && shiftPressed(window))
    {
        Camera& cam = app.renderState->camera;

        float panSpeed = app.zoom * 0.0015f;

        glm::vec3 right = cam.right;
        glm::vec3 up    = cam.up;

        cam.lookAt -= right * float(dx) * panSpeed;
        cam.lookAt += up    * float(dy) * panSpeed;

        cam.position -= right * float(dx) * panSpeed;
        cam.position += up    * float(dy) * panSpeed;

        app.camchanged = true;
    }
    // LEFT DRAG -> ORBIT
    else if (app.leftMousePressed)
    {
        app.phi   -= dx / app.width;
        app.theta -= dy / app.height;

        app.theta = glm::clamp(app.theta, 0.001f, PI - 0.001f);
        app.camchanged = true;
    }
    // RIGHT DRAG -> ZOOM (optional)
    else if (app.rightMousePressed)
    {
        app.zoom *= std::exp(float(dy) * 0.002f);
        app.zoom = glm::clamp(app.zoom, 0.1f, 1000.0f);
        app.camchanged = true;
    }
    // MIDDLE DRAG -> PAN (legacy support)
    else if (app.middleMousePressed)
    {
        Camera& cam = app.renderState->camera;

        float panSpeed = app.zoom * 0.0015f;

        glm::vec3 right = cam.right;
        glm::vec3 up    = cam.up;

        cam.lookAt -= right * float(dx) * panSpeed;
        cam.lookAt += up    * float(dy) * panSpeed;

        app.camchanged = true;
    }

end:
    app.lastX = xpos;
    app.lastY = ypos;
}


void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    AppState& app = AppState::Get();
    if (MouseOverImGuiWindow() || app.locked) return;

    // Sensitivity (tune this)
    const float zoomSpeed = 0.1f;

    // Trackpad-safe (continuous)
    app.zoom *= std::exp(-yoffset * zoomSpeed);

    // Clamp zoom
    app.zoom = glm::clamp(app.zoom, 0.1f, 1000.0f);

    app.camchanged = true;
}
