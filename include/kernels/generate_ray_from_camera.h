#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>

#include "sceneStructs.h"
#include "config.h"

__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* __restrict__ pathSegments, int* __restrict__ pathIndices);
