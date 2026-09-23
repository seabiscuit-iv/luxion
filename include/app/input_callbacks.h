#pragma once

// glew must precede glfw3
#include <GL/glew.h>
#include <GLFW/glfw3.h>

bool shiftPressed(GLFWwindow* window);
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
void processKeyboardMovement(GLFWwindow* window, float dt);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
void mousePositionCallback(GLFWwindow* window, double xpos, double ypos);
void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
