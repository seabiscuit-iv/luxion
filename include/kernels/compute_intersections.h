#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>

#include "sceneStructs.h"
#include "config.h"

__global__ void computeIntersections(
    int depth,
    int num_paths,
    const PathSegment* __restrict__ pathSegments,
    const int* __restrict__ pathIndices,
    const Geom* __restrict__ geoms,
    int geoms_size,
    ShadeableIntersection* __restrict__ intersections);
