#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

#include <glm/glm.hpp>

#include "sceneStructs.h"

class Scene;
class GuiDataContainer;
struct OptixAlphaMaterial;

struct PathTraceState {
    Scene* hst_scene = nullptr;
    GuiDataContainer* guiData = nullptr;
    glm::vec3* dev_image = nullptr;
    Geom* dev_geoms = nullptr;
    Material* dev_materials = nullptr;
    PathSegment* dev_paths = nullptr;
    ShadeableIntersection* dev_intersections = nullptr;
    ShadeableIntersection* dev_direct_light_intersections = nullptr;
    ShadeableIntersection* dev_environment_map_intersections = nullptr;

    int* dev_material_ids = nullptr; //for optix
    OptixAlphaMaterial* dev_alpha_materials = nullptr;

    glm::vec3** dev_vertex_buffer_locs = nullptr;
    Triangle** dev_triangle_buffer_locs = nullptr;
    glm::vec3** dev_normal_buffer_locs = nullptr;
    glm::vec2** dev_uv_buffer_locs = nullptr;

    // Optix
    CUdeviceptr d_optix_paramters = 0;

    uint32_t* dev_morton_codes = nullptr;
    int* dev_path_indices_A = nullptr;
    int* dev_path_indices_B = nullptr;
    int* dev_num_active_paths = nullptr;

    cudaArray_t dev_exr_array = nullptr;
    cudaTextureObject_t exr_texture = 0;

    // emissive area sampling stuff
    int* dev_emissive_geoms = nullptr;
    float* dev_emissive_geom_area_prefix = nullptr;

    // hdri sampling
    float* dev_hdri_marginal_cdf = nullptr;
    float* dev_hdri_conditional_cdfs = nullptr;

    static PathTraceState& Get() {
        static PathTraceState instance;
        return instance;
    }

    // Prevent copying and assignment
    PathTraceState(const PathTraceState&) = delete;
    PathTraceState& operator=(const PathTraceState&) = delete;
    PathTraceState() = default;
};

void InitDataContainer(GuiDataContainer* guiData);
