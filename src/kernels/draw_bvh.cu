#include "kernels/draw_bvh.h"

#include "intersections.h"
#include "common.h"

__global__ void drawBVH(
    int depth,
    int num_paths,
    PathSegment* __restrict__ pathSegments,
    const int* __restrict__ pathIndices,
    const Geom* __restrict__ geoms,
    int geoms_size,
    ShadeableIntersection* __restrict__ intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment &pathSegment = pathSegments[pathIndices[path_index]];

        int count = 0;

        for (int i = 0; i < geoms_size; i++)
        {
            const Geom &geom = geoms[i];

            if (geom.type == MESH)
            {
                count += bvhCountHits(geom, pathSegment.ray);
            }
        }

        pathSegment.color += float(count) * glm::vec3(0.001f);
    }
}
