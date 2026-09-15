#ifndef COOK_TORRANCE
#define COOK_TORRANCE

#include "common.h"
#include <cmath>

#include "sceneStructs.h"
#include "interactions.h"

#include <thrust/random.h>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#define INV_PI 0.3183098f

#define CLAMP_POS(x) glm::max(x, 0.0f)

// This is a workaround, and is not physically accurate. This should be replaced with direct light sampling and MIS
#define MICROFACET_REMOVE_FIREFLIES 0

namespace CookTorrance {

    __device__ float D_TrowbridgeReitz(glm::vec3 h, glm::vec3 n, float alpha);

    __device__ glm::vec3 F_SchlickApprox(float v_dot_h, const glm::vec3& f0);

    __device__ float Smith_GGX(glm::vec3 w, glm::vec3 n, float alpha);

    __device__ float Smith_G(glm::vec3 v, glm::vec3 l, glm::vec3 n, float alpha);
    
    __device__ glm::vec3 BRDF(glm::vec3 v, glm::vec3 n, glm::vec3 l, glm::vec3 albedo, float roughness, float metallic, float ior);

    __host__ __device__ glm::vec3 sphericalToCartesian(glm::vec3 spherical);

    __device__ void sampleGGX(PathSegment &path, int idx, int iter, int depth, glm::vec3 wo, glm::vec3 n, float roughness, thrust::default_random_engine &rng);

    __device__ void sampleCookTorrance(PathSegment &path, int idx, int iter, int depth, glm::vec3 wo, glm::vec3 n, float roughness, float metallic, thrust::default_random_engine &rng, glm::vec3 color, float ior);

    __device__ float PDF_GGX( glm::vec3 wo, glm::vec3 wi, glm::vec3 n, float roughness);
    
    __device__ float PDF(glm::vec3 wo, glm::vec3 wi, glm::vec3 n, float roughness, float metallic, glm::vec3 color, float ior);

    
    __device__ glm::vec3 shadePathCookTorrance(
        PathSegment &path,
        glm::vec3 albedo,
        glm::vec3 normal,
        glm::vec3 wi,
        float roughness,
        float metallic,
        float ior
    );

}

#endif // COOK_TORRANCE