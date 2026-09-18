#include "sample_materials.h"

#include "shaders/lambert.h"
#include "shaders/specular.h"
#include "shaders/cook_torrance.h"
#include "shaders/glass.h"

__device__ void sample_materials(
    const Material& material, 
    const ShadeableIntersection& intersection, 
    PathSegment& path, 
    int idx,
    int num_paths,
    int iter,
    int depth,
    thrust::default_random_engine& rng,
    glm::vec3 materialColor,
    glm::vec3 normal,
    float roughness,
    float metallic
) {
    #if UBER_SHADER
        CookTorrance::sampleCookTorrance(path, idx, iter, depth, -path.ray.direction, normal, roughness, metallic, rng, materialColor, material.indexOfRefraction, material.transmission);
    #else
        if (material.material_type == MaterialType::Emissive || material.material_type == MaterialType::Diffuse) {
            Lambert::sampleHemisphere(idx, num_paths, iter, depth, path, rng, normal);
        } 
        else if (material.material_type == MaterialType::Specular) {
            PerfectSpecular::sampleMirror(path, normal);
        }
        else if (material.material_type == MaterialType::Microfacet) {
            CookTorrance::sampleCookTorrance(path, idx, iter, depth, -path.ray.direction, normal, roughness, metallic, rng, materialColor, material.indexOfRefraction, material.transmission);
        }
        else if (material.material_type == MaterialType::Glass) {
            TransmissiveGlass::sampleGlass(path, material, rng, normal);
        }
    #endif
}
