#include "kernels/compute_intersections.h"

#include "intersections.h"
#include "common.h"

__global__ void computeIntersections(
    int depth,
    int num_paths,
    const PathSegment* __restrict__ pathSegments,
    const int* __restrict__ pathIndices,
    const Geom* __restrict__ geoms,
    int geoms_size,
    ShadeableIntersection* __restrict__ intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        const PathSegment pathSegment = pathSegments[pathIndices[path_index]];
        ShadeableIntersection isect = intersections[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        for (int i = 0; i < geoms_size; i++)
        {
            const Geom &geom = geoms[i];

            if (geom.type == CUBE)
            {
                // 94 vgprs
                #if ENABLE_BOX_INTERSECTION
                    t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
                #else 
                    t = -1.0f;
                #endif
            }
            else if (geom.type == SPHERE)
            {
                // 78 vgprs
                #if ENABLE_SPHERE_INTERSECTION
                    t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
                #else 
                    t = -1.0f;
                #endif
            }
            else if (geom.type == MESH)
            {
                // 80 VGPRs
                #if ENABLE_MESH_INTERSECTION
                    t = meshIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
                #else
                    t = -1.0f;
                #endif
            }

            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
            }
        }

        if (hit_geom_index == -1)
        {
            isect.t = -1.0f;
        }
        else
        {
            // The ray hits something
            isect.t = t_min;
            isect.materialId = geoms[hit_geom_index].materialid;
            isect.surfaceNormal = normal;
        }

        intersections[path_index] = isect;
    }
}
