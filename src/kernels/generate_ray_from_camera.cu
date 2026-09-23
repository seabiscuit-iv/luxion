#include "kernels/generate_ray_from_camera.h"

#include "utilities.h"
#include "common.h"

#include <thrust/random.h>

__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* __restrict__ pathSegments, int* __restrict__ pathIndices)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(0.0f);
        segment.throughput = glm::vec3(1.0f);
        segment.kill = false;

        
        CREATE_RANDOM_ENGINE(iter, index, traceDepth, u01, rng);

        float x1 = u01(rng) - 0.5f;
        float x2 = u01(rng) - 0.5f;

        float pX = (float(x) + x1 + 0.5f) - (float)cam.resolution.x * 0.5f;
        float pY = (float(y) + x2 + 0.5f) - (float)cam.resolution.y * 0.5f;

        segment.ray.direction = glm::normalize(
            cam.view 
            - (cam.right * cam.pixelLength.x * pX) 
            - (cam.up    * cam.pixelLength.y * pY)
        );
        segment.pixelIndex = index;
        pathIndices[index] = index;
    }
}
