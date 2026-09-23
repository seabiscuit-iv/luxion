#include "kernels/sample_direct_light.h"

#include "pathtrace/dev_options.h"
#include "common.h"
#include "utilities.h"

#include <thrust/random.h>

__global__ void sampleDirectLight(
    int iter,
    int num_paths,
    PathSegment* pathSegments,
    int depth,
    int num_emissive_geoms,
    int* emissive_geoms,
    float* emissive_geoms_area_prefix,
    const Geom* __restrict__ geoms,
    float total_emissive_mesh_area,
    float* marginal_cdf,
    float* conditional_cdfs,
    int exr_width,
    int exr_height
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths)
    {
        return;
    }

    PathSegment& path = pathSegments[idx];
    if (path.kill)
    {
        return;
    }

    thrust::default_random_engine rng = makeSeededRandomEngine(iter, path.pixelIndex, depth);
    
    if (DEV_OPTIONS.direct_light_sampling) {
        thrust::uniform_real_distribution<float> u01(0, 1);
        float rand = u01(rng);

        // binary search on dev_emissive_geom_area_prefix (range 0 .. num_emissive_geoms)
        int select = cudaUtils::select_from_cdf(emissive_geoms_area_prefix, num_emissive_geoms, rand);
        const Geom& emissive_geom = geoms[emissive_geoms[select]];

        float lower = (select == 0) ? 0.0f : emissive_geoms_area_prefix[select - 1];
        float upper = emissive_geoms_area_prefix[select];
        float denominator = upper - lower;
        float tri_offset = (denominator > 1e-10f) ? (rand - lower) / denominator : 0.0f;
        tri_offset = glm::clamp(tri_offset, 0.0f, 1.0f);

        int tri_select = cudaUtils::select_from_cdf(emissive_geom.mesh.d_triangle_area_percentage_prefix, emissive_geom.mesh.num_triangles, tri_offset);

        // sample the triangle at tri_select
        Triangle& tri = emissive_geom.mesh.d_triangles[tri_select];
        glm::vec3 v0 = emissive_geom.mesh.d_verts[tri.v_indices[0]];
        glm::vec3 v1 = emissive_geom.mesh.d_verts[tri.v_indices[1]];
        glm::vec3 v2 = emissive_geom.mesh.d_verts[tri.v_indices[2]];

        float r1 = sqrt(u01(rng));
        float r2 = u01(rng);
        float u = 1.0f - r1;
        float v = r2 * r1;

        glm::vec3 local_pos = u * v0 + v * v1 + (1.0f - u - v) * v2;
        glm::vec3 world_light_pos = glm::vec3(emissive_geom.transform * glm::vec4(local_pos, 1.0f));

        path.direct_light_sample = world_light_pos;
    }

    if (DEV_OPTIONS.environment_map_importance_sampling) {
        thrust::uniform_real_distribution<float> u01(0, 1);
        float rand = u01(rng);

        int marginal = cudaUtils::select_from_cdf(marginal_cdf, exr_height, rand);

        float lower = (marginal == 0) ? 0.0f : marginal_cdf[marginal - 1];
        float upper = marginal_cdf[marginal];
        float denominator = upper - lower;
        float offset = (denominator > 1e-10f) ? (rand - lower) / denominator : 0.0f;
        offset = glm::clamp(offset, 0.0f, 1.0f);

        int conditional = cudaUtils::select_from_cdf(conditional_cdfs + (marginal * exr_width), exr_width, offset);

        float u = (conditional + 0.5f) / exr_width;
        float v = (marginal + 0.5f) / exr_height;

        float phi = u * 2.0 * PI - PI;
        float theta = v * PI;

        glm::vec3 dir;
        float sin_theta = sinf(theta);
        dir.x = sin_theta * cosf(phi);
        dir.y = cosf(theta);
        dir.z = sin_theta * sinf(phi);

        path.environment_map_sample = dir;
    }
}
