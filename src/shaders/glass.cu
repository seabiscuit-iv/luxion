#include "shaders/glass.h"

namespace TransmissiveGlass
{
    __device__ glm::vec3 sampleSpecularTrans(glm::vec3 nor, glm::vec3 wo, float ior) {
        float etaA = 1.0f;
        float etaB = ior;
        
        bool enter = glm::dot(wo, nor) > 0.0f;

        float n12 = enter ? (etaA / etaB) : (etaB / etaA);
        
        glm::vec3 loc_normal = enter ? nor : -nor;

        glm::vec3 wi = glm::refract(-wo, loc_normal, n12);

        if (glm::dot(wi, wi) < 1e-6f || isnan(wi.x) || isnan(wi.y) || isnan(wi.z)) {
            return glm::reflect(-wo, loc_normal);
        }
    
        return glm::normalize(wi);
    }

    __device__ glm::vec3 sampleSpecularRefl(glm::vec3 nor, glm::vec3 wo) {
        glm::vec3 loc_normal = (glm::dot(wo, nor) > 0.0f) ? nor : -nor;
        return glm::reflect(-wo, loc_normal);
    }

    __device__ glm::vec3 FresnelDielectricEval(float cosThetaI, float ior) {
        float etaI = 1.;
        float etaT = ior;
        cosThetaI = glm::clamp(cosThetaI, -1.f, 1.f);

        bool enter = cosThetaI > 0.f;
        if (!enter) {
            float temp = etaI;
            etaI = etaT;
            etaT = temp;
            cosThetaI = glm::abs(cosThetaI);
        }

        //snells law
        float sin_t_i = glm::sqrt(glm::max(0.0f, 1.0f - cosThetaI * cosThetaI));
        float sin_t_t = etaI / etaT * sin_t_i;
        if (sin_t_t >= 1) {
            return glm::vec3(1);
        }

        float cosThetaT = glm::sqrt(glm::max(0.0f, 1.0f - sin_t_t * sin_t_t));

        float r_parallel = ((etaT * cosThetaI) - (etaI * cosThetaT)) / ((etaT * cosThetaI) + (etaI * cosThetaT));
        float r_perpendicular = ((etaI * cosThetaI) - (etaT * cosThetaT)) / ((etaI * cosThetaI) + (etaT * cosThetaT));
        return glm::vec3(r_parallel * r_parallel + r_perpendicular * r_perpendicular) / 2.0f;
    }

    __device__ void sampleGlass(PathSegment &path, const Material &material, thrust::default_random_engine &rng, glm::vec3 normal) {
        glm::vec3 wo = -path.ray.direction;

        thrust::uniform_real_distribution<float> u01(0, 1);
        float r = u01(rng);

        float cos_theta_I = glm::dot(wo, normal);
        glm::vec3 F_vec = FresnelDielectricEval(cos_theta_I, material.indexOfRefraction);
        float F = glm::clamp(F_vec.r, 0.0f, 1.0f);

        glm::vec3 wi = glm::vec3(0.0, 1.0, 0.0);
        if (r < F) {
            wi = sampleSpecularRefl(normal, wo);
        }   
        else {
            wi = sampleSpecularTrans(normal, wo, material.indexOfRefraction);
        }

        // the NaN is somewhere here

        path.sample_dir = wi;
    }

    __device__ void shadePathGlass(
        PathSegment &path,  
        const Material &material,
        glm::vec3 color
    ) {
        path.throughput *= glm::mix(glm::vec3(1.0), color, material.alpha);
    }
}