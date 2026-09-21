#include <optix.h>

#include <sutil/vec_math.h>
#include <sutil/helpers.h>

#include "optix_triangle_params.cuh"

extern "C" {
__constant__ Params params;
}


static __forceinline__ __device__ void computeRay( uint3 idx, uint3 dim, float3& origin, float3& direction, float3& directlight_dir, float3& envmap_sample )
{

    OptixPathSegment& path_segment = params.path_segments[idx.x];

    origin = path_segment.ray.origin;
    direction = path_segment.ray.direction;
    directlight_dir = path_segment.direct_light_sample;
    envmap_sample = path_segment.environment_map_sample;
}


static __forceinline__ __device__ float2 interpolate_uv( const float2* uv_buffer, const OptixTriangle& triangle, float2 barycentrics )
{
    if (uv_buffer == nullptr) {
        return make_float2(0.0f, 0.0f);
    }

    float2 A_uv = uv_buffer[triangle.uv_indices[0]];
    float2 B_uv = uv_buffer[triangle.uv_indices[1]];
    float2 C_uv = uv_buffer[triangle.uv_indices[2]];

    float bA = 1.0f - barycentrics.x - barycentrics.y;

    return bA * A_uv + barycentrics.x * B_uv + barycentrics.y * C_uv;
}


static __forceinline__ __device__ unsigned int hash_u32( unsigned int x )
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}


extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    if ( params.path_segments[idx.x].kill )
    {
        params.shadeable_intersections[idx.x].t = -1.0f;
        params.direct_light_intersections[idx.x].t = -1.0f;
        params.environment_map_intersections[idx.x].t = -0.5f;
        return;
    }

    float3 ray_origin, ray_direction, direct_light_sample, environment_map_sample;
    computeRay( idx, dim, ray_origin, ray_direction, direct_light_sample, environment_map_sample );

    optixTrace(
        params.handle,
        ray_origin,
        ray_direction,
        0.0f,
        1e16f,
        0.0f,
        OptixVisibilityMask( 255 ),
        OPTIX_RAY_FLAG_NONE,
        RAY_TYPE_RADIANCE,
        RAY_TYPE_COUNT,
        RAY_TYPE_RADIANCE
    );

    const OptixShadeableIntersection& intersection = params.shadeable_intersections[idx.x];
    if (intersection.t > 0.0f)
    {
        float3 hit_point = ray_origin + intersection.t * ray_direction;
        const float epsilon = 1e-4f;
        float3 shadow_ray_origin = hit_point + (intersection.surfaceNormal * epsilon);

        if (params.direct_light_sampling)
        {
            float3 ray_directlight_dir = direct_light_sample - shadow_ray_origin;
            float dist_to_light = length(ray_directlight_dir);
            ray_directlight_dir = ray_directlight_dir / dist_to_light; // Normalize

            optixTrace(
                params.handle,
                shadow_ray_origin,
                ray_directlight_dir,
                0.0f,
                dist_to_light * 1.0001f + epsilon,
                0.0f,
                OptixVisibilityMask( 255 ),
                OPTIX_RAY_FLAG_NONE,
                RAY_TYPE_DIRECT_LIGHT,
                RAY_TYPE_COUNT,
                RAY_TYPE_DIRECT_LIGHT
            );
        }
        else {
            params.direct_light_intersections[idx.x].t = -1.0f;
        }

        if (params.envmap_sampling)
        {
            unsigned int occluded = 1;
            optixTrace(
                params.handle,
                shadow_ray_origin,
                environment_map_sample,
                0.0f,
                1e16f,
                0.0f,
                OptixVisibilityMask( 255 ),
                OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                RAY_TYPE_ENV_MAP,
                RAY_TYPE_COUNT,
                RAY_TYPE_ENV_MAP,
                occluded
            );

            params.environment_map_intersections[idx.x].t = occluded ? 1.0f : -1.0f;
        }
        else {
            params.environment_map_intersections[idx.x].t = 1.0f;
        }
    }
    else {
        params.direct_light_intersections[idx.x].t = -1.0f;

        // printf("1\n");
        params.environment_map_intersections[idx.x].t = -0.5f;
    }
}


extern "C" __global__ void __miss__ms()
{
    const uint3 idx = optixGetLaunchIndex();
    const OptixPathSegment& path_segment = params.path_segments[idx.x];
    OptixShadeableIntersection& shadeable_intersection = params.shadeable_intersections[idx.x];

    shadeable_intersection.t = -1.0f;
}


extern "C" __global__ void __closesthit__ch()
{
    const float2 barycentrics = optixGetTriangleBarycentrics();
    const unsigned int object_ID = optixGetInstanceId();
    const unsigned int prim_ID = optixGetPrimitiveIndex();

    const uint3 idx = optixGetLaunchIndex();
    const OptixPathSegment& path_segment = params.path_segments[idx.x];
    OptixShadeableIntersection& shadeable_intersection = params.shadeable_intersections[idx.x];

    int material_id = params.material_ids[object_ID];

    float3* vertex_buffer = params.vertex_buffer_locations[object_ID];
    OptixTriangle* triangle_buffer = params.triangle_buffer_locations[object_ID];
    float3* normal_buffer = params.normal_buffer_locations[object_ID];
    float2* uv_buffer = params.uv_buffer_locations[object_ID];

    OptixTriangle& triangle = triangle_buffer[prim_ID];

    float3 A = vertex_buffer[triangle.v_indices[0]];
    float3 B = vertex_buffer[triangle.v_indices[1]];
    float3 C = vertex_buffer[triangle.v_indices[2]];

    float3 edge1 = B - A;
    float3 edge2 = C - A;

    float3 normal = normalize(cross(edge1, edge2));

    if (normal_buffer != nullptr) {
        A = normal_buffer[triangle.n_indices[0]];
        B = normal_buffer[triangle.n_indices[1]];
        C = normal_buffer[triangle.n_indices[2]];

        float bA = 1.0f - barycentrics.x - barycentrics.y;

        normal = normalize(bA * A + barycentrics.x * B + barycentrics.y * C);
    }

    float3 tangent = make_float3(1.0, 0.0, 0.0);

    float2 uv = make_float2(0.0f, 0.0f);
    if (uv_buffer != nullptr) {
        float2 A_uv = uv_buffer[triangle.uv_indices[0]];
        float2 B_uv = uv_buffer[triangle.uv_indices[1]];
        float2 C_uv = uv_buffer[triangle.uv_indices[2]];

        float bA = 1.0f - barycentrics.x - barycentrics.y;

        uv = bA * A_uv + barycentrics.x * B_uv + barycentrics.y * C_uv;

        // tangent calculation
        float2 duv1 = B_uv - A_uv;
        float2 duv2 = C_uv - A_uv;

        float det = (duv1.x * duv2.y - duv2.x * duv1.y);

        float f = (fabsf(det) < 1e-10f) ? 0.0f : 1.0f / det;

        if (f == 0.0f) {
            tangent = make_float3(1.0f, 0.0f, 0.0f); 
        }
        else {
            tangent.x = f * (duv2.y * edge1.x - duv1.y * edge2.x);
            tangent.y = f * (duv2.y * edge1.y - duv1.y * edge2.y);
            tangent.z = f * (duv2.y * edge1.z - duv1.y * edge2.z);
        }
    }

    normal = normalize(optixTransformNormalFromObjectToWorldSpace(normal));
    tangent = normalize(optixTransformVectorFromObjectToWorldSpace(tangent));

    float3 T = tangent - dot(tangent, normal) * normal;
    float len2 = dot(T, T);
    if (len2 > 1e-10f) {
        tangent = T * rsqrtf(len2);
    } else {
        tangent = (fabsf(normal.x) > 0.9f) ? make_float3(0, 1, 0) : make_float3(1, 0, 0);
        tangent = normalize(cross(tangent, normal));
    }

    shadeable_intersection.materialId = material_id;
    shadeable_intersection.t = optixGetRayTmax();
    shadeable_intersection.u = uv.x;
    shadeable_intersection.v = uv.y;

    shadeable_intersection.surfaceNormal = normal;
    shadeable_intersection.surfaceTangent = tangent;
}



extern "C" __global__ void __miss__ms_direct_light()
{
    const uint3 idx = optixGetLaunchIndex();
    const OptixPathSegment& path_segment = params.path_segments[idx.x];
    OptixShadeableIntersection& shadeable_intersection = params.direct_light_intersections[idx.x];

    shadeable_intersection.t = -1.0f;
}




extern "C" __global__ void __closesthit__ch_direct_light()
{
    const float2 barycentrics = optixGetTriangleBarycentrics();
    const unsigned int object_ID = optixGetInstanceId();
    const unsigned int prim_ID = optixGetPrimitiveIndex();

    const uint3 idx = optixGetLaunchIndex();
    OptixShadeableIntersection& shadeable_intersection = params.direct_light_intersections[idx.x];

    int material_id = params.material_ids[object_ID];

    float3* vertex_buffer = params.vertex_buffer_locations[object_ID];
    OptixTriangle* triangle_buffer = params.triangle_buffer_locations[object_ID];
    float3* normal_buffer = params.normal_buffer_locations[object_ID];
    float2* uv_buffer = params.uv_buffer_locations[object_ID];

    OptixTriangle& triangle = triangle_buffer[prim_ID];

    float bA = 1.0f - barycentrics.x - barycentrics.y;

    float3 normal;
    if (normal_buffer != nullptr) {
        float3 A = normal_buffer[triangle.n_indices[0]];
        float3 B = normal_buffer[triangle.n_indices[1]];
        float3 C = normal_buffer[triangle.n_indices[2]];

        normal = normalize(bA * A + barycentrics.x * B + barycentrics.y * C);
    }
    else {
        float3 A = vertex_buffer[triangle.v_indices[0]];
        float3 B = vertex_buffer[triangle.v_indices[1]];
        float3 C = vertex_buffer[triangle.v_indices[2]];

        normal = normalize(cross(B - A, C - A));
    }

    float2 uv = interpolate_uv(uv_buffer, triangle, barycentrics);

    normal = normalize(optixTransformNormalFromObjectToWorldSpace(normal));

    shadeable_intersection.materialId = material_id;
    shadeable_intersection.t = optixGetRayTmax();
    shadeable_intersection.u = uv.x;
    shadeable_intersection.v = uv.y;

    shadeable_intersection.surfaceNormal = normal;
}




extern "C" __global__ void __miss__ms_envmap()
{
    optixSetPayload_0( 0 );
}



extern "C" __global__ void __anyhit__ms_all()
{
    const unsigned int object_ID = optixGetInstanceId();
    const unsigned int prim_ID = optixGetPrimitiveIndex();

    const OptixAlphaMaterial& material = params.alpha_materials[params.material_ids[object_ID]];

    if (material.alpha_mode == OPTIX_ALPHA_MODE_OPAQUE) {
        return;
    }

    float alpha = material.alpha;

    if (material.albedo_tex != 0) {
        const OptixTriangle& triangle = params.triangle_buffer_locations[object_ID][prim_ID];
        float2 uv = interpolate_uv(params.uv_buffer_locations[object_ID], triangle, optixGetTriangleBarycentrics());

        uv = uv * material.tex_scale;
        if (material.tex_rotation != 0.0f) {
            float c = cosf(material.tex_rotation);
            float s = sinf(material.tex_rotation);
            uv = make_float2(c * uv.x - s * uv.y, s * uv.x + c * uv.y);
        }
        uv = uv + material.tex_offset;

        float4 tex = tex2D<float4>(material.albedo_tex, uv.x, uv.y);
        alpha *= tex.w;
    }

    if (material.alpha_mode == OPTIX_ALPHA_MODE_MASK) {
        if (alpha < material.alpha_cutoff) {
            optixIgnoreIntersection();
        }
        return;
    }

    const float3 dir = optixGetWorldRayDirection();

    unsigned int h = hash_u32(optixGetLaunchIndex().x ^ hash_u32(params.iteration));
    h = hash_u32(h ^ __float_as_uint(dir.x));
    h = hash_u32(h ^ __float_as_uint(dir.y));
    h = hash_u32(h ^ __float_as_uint(dir.z));
    h = hash_u32(h ^ object_ID);
    h = hash_u32(h ^ prim_ID);

    float rnd = (h >> 8) * (1.0f / 16777216.0f);
    if (rnd >= alpha) {
        optixIgnoreIntersection();
    }
}

