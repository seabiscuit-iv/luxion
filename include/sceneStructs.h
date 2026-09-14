#pragma once

#include <cuda_runtime.h>

#include "config.h"
#include "glm/glm.hpp"
#include "mesh.h"

#include <string>
#include <vector>

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType
{
    SPHERE,
    CUBE,
    MESH
};

enum MaterialType {
#if UBER_SHADER
    Microfacet
#else
    Diffuse = 0, // lambertian perfect diffuse
    Specular, // perfectly specular
    Emissive,
    Microfacet,
    Glass
#endif
};


struct Triangle {
    uint32_t v_indices[3];
    uint32_t n_indices[3];
    uint32_t uv_indices[3];

    Triangle(int v[3], int n[3], int uv[3])
    {
        for(int i = 0; i < 3; i++) {
            v_indices[i] = v[i];
            n_indices[i] = n[i];
            uv_indices[i] = uv[i];
        }
    }

    glm::vec3 centroid(const std::vector<glm::vec3> &verts) const {
        glm::vec3 centroid(0.0f);
        centroid += verts[v_indices[0]];
        centroid += verts[v_indices[1]];
        centroid += verts[v_indices[2]];
        centroid /= 3.0f;
        return centroid;
    }
};

struct Ray
{ 
    glm::vec3 origin;
    glm::vec3 direction;
    glm::vec3 inv_direction;
    glm::ivec3 sign;
};

struct Geom
{
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;

    // these are for the mesh, they will be null otherwise
    Mesh mesh;
};


struct TextureTransform {
    glm::vec2 offset{0.0f, 0.0f};
    glm::vec2 scale{1.0f, 1.0f};
    float rotation = 0.0f;
};

struct Material
{
    MaterialType material_type;
    glm::vec3 color;
    int albedo_tex;
    struct
    {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    
    struct {
        float emission_strength = 0.0f;
        glm::vec3 emission_color;
        int emissive_tex = -1;
        TextureTransform emissive_tex_transform;
    } emission;

    float roughness = 0.0f;
    float metallic = 0.0f;
    float alpha = 1.0f;
    int normal_tex;
    int metallic_roughness_tex = -1;

    TextureTransform albedo_tex_transform;

    TextureTransform normal_tex_transform;

    TextureTransform metallic_roughness_tex_transform;
};

struct Camera
{
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
};

struct RenderState
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment
{
    Ray ray;
    glm::vec3 color;
    glm::vec3 throughput;
    glm::vec3 sample_dir; // brdf sample dir
    glm::vec3 direct_light_sample; // direct light sample dir
    glm::vec3 environment_map_sample;
    int pixelIndex;
    bool kill = false;
    bool last_bounce_was_specular = false;
    float last_pdf = -1.0f;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
  float t;                  // 0
  glm::vec3 surfaceNormal;  // 4
  glm::vec3 surfaceTangent;  // 4
  int materialId;           // 16
  float _pad0;              // 20 (This fixes your Offset 3 mismatch)
  glm::vec2 uvs;            // 24
}; // Total size should be 32. No alignas(16) needed if manually padded.