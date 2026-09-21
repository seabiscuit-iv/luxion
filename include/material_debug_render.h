#pragma once 

#include <cuda.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>

#include "sceneStructs.h"
#include "config.h"

__device__ void render_material_debug_mode (
    PathSegment& path, 
    glm::vec3 materialColor, 
    glm::vec3 normal, 
    glm::vec3 normal_map,
    float roughness,
    float metallic,
    float alpha,
    int material_debug_mode
);
