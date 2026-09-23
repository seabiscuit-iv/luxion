#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>

#include "sceneStructs.h"
#include "config.h"

__global__ void intersectionPrecompute(int n, PathSegment* __restrict__ pathSegments, const int* __restrict__ pathIndices, const Geom* mesh, uint32_t* morton_codes);
