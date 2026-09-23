#pragma once

enum RayType
{
    RAY_TYPE_RADIANCE = 0,
    RAY_TYPE_DIRECT_LIGHT,
    RAY_TYPE_ENV_MAP,
    RAY_TYPE_COUNT
};

// struct Params {
//     float3 cam_eye;
//     float4 InvViewProj[4];
//     OptixTraversableHandle as_handle;
// };

struct OptixRay
{ 
    float3 origin;
    float3 direction;
    float3 inv_direction;
    int3 sign;
};

struct OptixPathSegment
{
    OptixRay ray;
    float3 color;
    float3 throughput;
    float3 sample_dir;
    float3 direct_light_sample;
    float3 environment_map_sample;
    int pixelIndex;
    bool kill;
    bool last_bounce_was_specular;
    float last_pdf;
};

struct OptixShadeableIntersection
{
  float t;                  // 0
  float3 surfaceNormal;     // 4 (CUDA float3 at offset 4 is fine here)
  float3 surfaceTangent;
  int materialId;           // 16
  float _pad0;              // 20
  float u;               // 24
  float v;                  
};

// sceneStructs.h
enum OptixAlphaMode
{
    OPTIX_ALPHA_MODE_OPAQUE = 0,
    OPTIX_ALPHA_MODE_MASK,
    OPTIX_ALPHA_MODE_BLEND
};

struct OptixAlphaMaterial
{
    int alpha_mode;
    float alpha;
    float alpha_cutoff;
    unsigned long long albedo_tex;
    float2 tex_offset;
    float2 tex_scale;
    float tex_rotation;
};

struct OptixTriangle {
    unsigned int v_indices[3];
    unsigned int n_indices[3];
    unsigned int uv_indices[3];
};


struct Params
{
    OptixTraversableHandle handle;
    OptixPathSegment* path_segments;
    int* path_indices;
    float3* debug_image;
    OptixShadeableIntersection* shadeable_intersections;    
    OptixShadeableIntersection* direct_light_intersections;    
    OptixShadeableIntersection* environment_map_intersections;    
    int* material_ids;
    OptixAlphaMaterial* alpha_materials;
    unsigned int iteration;

    float3** vertex_buffer_locations;
    OptixTriangle** triangle_buffer_locations;
    float3** normal_buffer_locations;
    float2** uv_buffer_locations;

    bool direct_light_sampling;
    bool envmap_sampling;
};

struct RayGenData
{
    // No data needed
};


struct MissData
{
    float3 bg_color;
};


struct HitGroupData
{
    // No data needed
};
