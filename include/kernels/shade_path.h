#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>

#include "sceneStructs.h"
#include "texture.h"
#include "config.h"

__global__ void shadePath(
    int iter,
    int num_paths,
    PathSegment* __restrict__ pathSegments,
    const int* __restrict__ pathIndices,
    int* __restrict__ nextPathIndices,
    int* __restrict__ numNextPaths,
    Material* __restrict__ materials,
    ShadeableIntersection* __restrict__ shadeableIntersections,
    ShadeableIntersection* __restrict__ directLightIntersections,
    ShadeableIntersection* __restrict__ environmentMapIntersections,
    int depth,
    bool has_exr,
    cudaTextureObject_t exr,
    TextureData* textures,
    int num_emissive_geoms,
    int* emissive_geoms,
    float* emissive_geoms_area_prefix,
    const Geom* __restrict__ geoms,
    float total_emissive_mesh_area,
    float* marginal_cdf,
    float* conditional_cdfs,
    int exr_width,
    int exr_height
);
