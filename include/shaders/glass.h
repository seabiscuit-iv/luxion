#ifndef GLASS_MATERIAL
#define GLASS_MATERIAL

#include "common.h"
#include <cmath>

#include "sceneStructs.h"
#include "interactions.h"

#include <thrust/random.h>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#define INV_PI 0.3183098f

namespace TransmissiveGlass
{
    __device__ glm::vec3 sampleSpecularTrans(glm::vec3 nor, glm::vec3 wo, float ior);

    __device__ glm::vec3 sampleSpecularRefl(glm::vec3 nor, glm::vec3 wo);

    __device__ glm::vec3 FresnelDielectricEval(float cosThetaI, float ior);

    __device__ void sampleGlass(PathSegment &path, const Material &material, thrust::default_random_engine &rng, glm::vec3 normal);

    __device__ void shadePathGlass(
        PathSegment &path,  
        const Material &material,
        glm::vec3 color
    );
}

#endif // GLASS_MATERIAL