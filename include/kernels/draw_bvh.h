#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>

#include "sceneStructs.h"
#include "config.h"

__global__ void drawBVH(
    int depth,
    int num_paths,
    PathSegment* __restrict__ pathSegments,
    const int* __restrict__ pathIndices,
    const Geom* __restrict__ geoms,
    int geoms_size,
    ShadeableIntersection* __restrict__ intersections);
