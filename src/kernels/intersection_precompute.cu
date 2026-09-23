#include "kernels/intersection_precompute.h"

#include "morton_codes.h"
#include "common.h"

__global__ void intersectionPrecompute(int n, PathSegment* __restrict__ pathSegments, const int* __restrict__ pathIndices, const Geom* mesh, uint32_t* morton_codes) {
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < n)
    {
        Ray r = pathSegments[pathIndices[path_index]].ray;

        r.origin = glm::vec3(mesh->inverseTransform * glm::vec4(r.origin, 1.0f));
        r.direction = glm::vec3(mesh->inverseTransform * glm::vec4(r.direction, 0.0f));

        r.inv_direction.x = __frcp_rn(r.direction.x);
        r.inv_direction.y = __frcp_rn(r.direction.y);
        r.inv_direction.z = __frcp_rn(r.direction.z);

        r.sign.x = (r.inv_direction.x < 0.0f) ? 1 : 0;
        r.sign.y = (r.inv_direction.y < 0.0f) ? 1 : 0;
        r.sign.z = (r.inv_direction.z < 0.0f) ? 1 : 0;

        BoundingBox bbox = mesh->mesh.bvh.dev_bvh[0].box;

        float scene_extent = glm::length(bbox.box_max - bbox.box_min);

        float t;
        morton_codes[path_index] = bbox.RayBoxInterection(r, t)
            ? rayMortonCode(r, scene_extent)
            : MORTON_CODE_MISS;
    }
}
