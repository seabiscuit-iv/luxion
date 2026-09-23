#pragma once

#include <string>

#include <glm/glm.hpp>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "sceneStructs.h"
#include "utilities.h"

class Scene;
struct ImGuiIO;

struct AppState {
    std::string startTimeString;
    std::string outputName; // if non-empty, overrides the default output image name (without .png)

    // For camera controls
    bool leftMousePressed = false;
    bool rightMousePressed = false;
    bool middleMousePressed = false;
    double lastX = 0.0;
    double lastY = 0.0;

    bool locked = false;
    bool hideImGui = false;
    bool showAnalytics = true;
    bool showStageTimings = false;

    bool camchanged = true;
    float dtheta = 0, dphi = 0;
    glm::vec3 cammove = glm::vec3(0.0f);
    glm::vec3 refUp = glm::vec3(0.0f);

    float zoom = 0.0f, theta = 0.0f, phi = 0.0f;
    glm::vec3 ogLookAt = glm::vec3(0.0f); // for recentering the camera

    Scene* scene = nullptr;
    GuiDataContainer* guiData = nullptr;
    RenderState* renderState = nullptr;
    int iteration = 0;

    int width = 0;
    int height = 0;

    GLuint positionLocation = 0;
    GLuint texcoordsLocation = 1;
    GLuint pbo = 0;
    GLuint displayImage = 0;

    GLFWwindow* window = nullptr;
    GuiDataContainer* imguiData = NULL;
    ImGuiIO* io = nullptr;
    bool mouseOverImGuiWinow = false;

    static AppState& Get() {
        static AppState instance;
        return instance;
    }

    // Prevent copying and assignment
    AppState(const AppState&) = delete;
    AppState& operator=(const AppState&) = delete;
    AppState() = default;
};
