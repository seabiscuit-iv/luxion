#include "shaders/cook_torrance.h"

#define EPSILON 1e-30f
#define PDF_CLAMP 1e-5f

namespace CookTorrance {

    __device__ float D_TrowbridgeReitz(glm::vec3 h, glm::vec3 n, float alpha) {
        float n_dot_h = glm::dot(n, h);

        if (n_dot_h <= 0.0f) {
            return 0.0f;
        }

        float alpha_sq = alpha * alpha;

        float numerator = alpha_sq;

        float n_dot_h_sq = n_dot_h * n_dot_h;
        float denom_component = ( n_dot_h_sq * alpha_sq ) + ( 1.0f - n_dot_h_sq );

        float denominator = denom_component * denom_component * glm::pi<float>();

        return numerator / glm::max(denominator, EPSILON);
    }

    __device__ glm::vec3 F_SchlickApprox(float v_dot_h, const glm::vec3& f0) {
        float m = 1.0f - v_dot_h;
        float m2 = m * m;
        return f0 + ( (glm::vec3(1.0f) - f0) * (m2 * m2 * m) );
    }


    __device__ float FresnelDielectric(float cosThetaI, float etaI, float etaT) {
        cosThetaI = glm::clamp(cosThetaI, 0.0f, 1.0f);

        float sinThetaI = glm::sqrt(glm::max(0.0f, 1.0f - cosThetaI * cosThetaI));
        float sinThetaT = (etaI / etaT) * sinThetaI;

        if (sinThetaT >= 1.0f) {
            return 1.0f;
        }

        float cosThetaT = glm::sqrt(glm::max(0.0f, 1.0f - sinThetaT * sinThetaT));

        float r_parallel = ((etaT * cosThetaI) - (etaI * cosThetaT)) / ((etaT * cosThetaI) + (etaI * cosThetaT));
        float r_perpendicular = ((etaI * cosThetaI) - (etaT * cosThetaT)) / ((etaI * cosThetaI) + (etaT * cosThetaT));

        return (r_parallel * r_parallel + r_perpendicular * r_perpendicular) / 2.0f;
    }

    __device__ glm::vec3 transmissionHalfVector(glm::vec3 wo, glm::vec3 wi, float etaI, float etaT) {
        return glm::normalize(-(etaI * wo + etaT * wi));
    }

    __device__ float Smith_GGX(glm::vec3 w, glm::vec3 n, float alpha) {
        float alpha_sq = alpha * alpha;
        float n_dot_w = CLAMP_POS(glm::dot(n, w));

        float numerator = 2.0f * n_dot_w;

        float under_sqrt = alpha_sq + (1.0f - alpha_sq) * (n_dot_w * n_dot_w);
        float denominator = n_dot_w + glm::sqrt(under_sqrt);

        return numerator / glm::max(denominator, EPSILON);
    }

    __device__ float Smith_G(glm::vec3 v, glm::vec3 l, glm::vec3 n, float alpha) {
        return Smith_GGX(v, n, alpha) * Smith_GGX(l, n, alpha);
    }
    
    __device__ glm::vec3 Fresnel(float cosTheta, glm::vec3 albedo, float metallic, float etaI, float etaT) {
        glm::vec3 dielectric = glm::vec3(FresnelDielectric(cosTheta, etaI, etaT));
        glm::vec3 conductor = F_SchlickApprox(cosTheta, albedo);

        return glm::mix(dielectric, conductor, metallic);
    }

    __device__ glm::vec3 BRDF(glm::vec3 v, glm::vec3 n, glm::vec3 l, glm::vec3 albedo, float roughness, float metallic, float ior, float transmission) {
        ior = glm::max(ior, 1.0001f);

        bool enter = glm::dot(v, n) > 0.0f;

        glm::vec3 n_facing = enter ? n : -n;

        float etaI = enter ? 1.0f : ior;
        float etaT = enter ? ior : 1.0f;

        glm::vec3 h = glm::normalize(v + l);

        roughness = glm::clamp(roughness, 0.0001f, 1.0f);
        float alpha = roughness * roughness;

        float D = D_TrowbridgeReitz(h, n_facing, alpha);
        glm::vec3 F = Fresnel(CLAMP_POS(glm::dot(v, h)), albedo, metallic, etaI, etaT);
        float G = Smith_G(v, l, n_facing, alpha);

        glm::vec3 numerator = D * F * G;
        float denominator = 4 * CLAMP_POS(glm::dot(n_facing, v)) * CLAMP_POS(glm::dot(n_facing, l));

        glm::vec3 specular = numerator / glm::max(denominator, EPSILON);

        glm::vec3 nonSpecular = glm::vec3(1.0f) - F; // Use the same F calculated for specular
        glm::vec3 k_S = F; // Specular ratio
        glm::vec3 k_D = (1.0f - metallic) * nonSpecular * (1.0f - transmission); // Diffuse ratio, must be 0 for full metallic

        glm::vec3 diffuse = k_D * albedo * INV_PI;

        return diffuse + specular;
    }

    __device__ glm::vec3 BTDF(glm::vec3 wo, glm::vec3 n, glm::vec3 wi, glm::vec3 albedo, float roughness, float metallic, float ior, float transmission)
    {
        float cosThetaI = glm::dot(wo, n);
        bool enter = cosThetaI > 0.f;

        float etaI = enter ? 1.0f : ior; // n_o
        float etaT = enter ? ior : 1.0f; // n_i

        glm::vec3 h_t = transmissionHalfVector(wo, wi, etaI, etaT);

        float jacobian_denom = (etaT * glm::dot(wi, h_t) + etaI * glm::dot(wo, h_t));
        float jacobian = (etaI * etaI) / glm::max(jacobian_denom * jacobian_denom, EPSILON);

        float conversion_term_numerator = glm::abs(glm::dot(wi, h_t)) * glm::abs(glm::dot(wo, h_t));
        float conversion_term_denominator = glm::abs(glm::dot(wi, n)) * glm::abs(glm::dot(wo, n));

        roughness = glm::clamp(roughness, 0.0001f, 1.0f);
        float alpha = roughness * roughness;

        float D = D_TrowbridgeReitz(h_t, n, alpha);
        float F = FresnelDielectric(glm::abs(glm::dot(wo, h_t)), etaI, etaT);

        bool G_vis = glm::dot(wi, h_t) * glm::dot(wi, n) > 0.0f && glm::dot(wo, h_t) * glm::dot(wo, n) > 0.0f;

        // Smith_G but for transmission
        float G = G_vis ? Smith_GGX(glm::dot(wi, n) > 0.0f ? wi : -wi, n, alpha) * Smith_GGX(glm::dot(wo, n) > 0.0f ? wo : -wo, n, alpha) : 0.0f;

        float DFG = D * G * (1.0f - F);

        return albedo * (transmission * (1.0f - metallic) * (conversion_term_numerator / conversion_term_denominator) * jacobian * DFG);
    }

    __host__ __device__ glm::vec3 sphericalToCartesian(glm::vec3 spherical) {
        float r = spherical.x;
        float theta = spherical.y;
        float phi = spherical.z;

        float x = r * glm::sin(theta) * glm::cos(phi);
        float y = r * glm::sin(theta) * glm::sin(phi);
        float z = r * glm::cos(theta);

        return glm::vec3(x, y, z);
    } 

    __device__ void sampleCookTorrance(PathSegment &path, int idx, int iter, int depth, glm::vec3 wo, glm::vec3 n, float roughness, float metallic, thrust::default_random_engine &rng, glm::vec3 color, float ior, float transmission) {
        thrust::uniform_real_distribution<float> u01(0, 1);
        float r = u01(rng);

        ior = glm::max(ior, 1.0001f);

        roughness = glm::clamp(roughness, 0.0001f, 1.0f);
        float alpha = roughness * roughness;

        bool enter = glm::dot(wo, n) > 0.0f;

        glm::vec3 n_facing = enter ? n : -n;

        float etaI = enter ? 1.0f : ior;
        float etaT = enter ? ior : 1.0f;

        float cos_macro = glm::abs(glm::dot(wo, n));

        glm::vec3 F_approx = Fresnel(cos_macro, color, metallic, etaI, etaT);
        float specularRatio = glm::clamp((F_approx.r + F_approx.g + F_approx.b) / 3.0f, 0.01f, 0.99f);

        float F_macro = FresnelDielectric(cos_macro, etaI, etaT);

        float probRefract = transmission * (1.0f - metallic) * (1.0f - F_macro);
        float probDiffuse = (1.0f - probRefract) * (1.0f - specularRatio) * (1.0f - metallic);
        float probSpecular = 1.0f - probRefract - probDiffuse;

        // diffuse lobe uses macronormal
        if (r > probSpecular && r <= probSpecular + probDiffuse) {
            path.sample_dir = calculateRandomDirectionInHemisphere(n_facing, rng);
            return;
        }

        // specular (perfect reflection) and refract (perfect refraction) uses facetnormal
        float x1 = u01(rng);
        float x2 = u01(rng);

        float theta_h = glm::atan((alpha * glm::sqrt(x1)) / glm::sqrt(1.0f - x1));
        float phi_h = 2.0f * PI * x2;

        glm::vec3 h = glm::normalize(sphericalToCartesian(glm::vec3(1.0, theta_h, phi_h)));

        glm::vec3 t = glm::normalize( abs(n_facing.z) < 0.999f ? glm::cross(glm::vec3(0,0,1), n_facing) : glm::cross(glm::vec3(1,0,0), n_facing) );
        glm::vec3 b = glm::cross(n_facing, t);

        glm::vec3 h_world = h.x * t + h.y * b + h.z * n_facing;

        if (r <= probSpecular) {
            glm::vec3 wi = glm::reflect(-wo, h_world);

            if (glm::dot(wi, n_facing) <= 0.0f) {
                path.sample_dir = glm::vec3(0.0f);
                return;
            }

            path.sample_dir = glm::normalize(wi);
            return;
        }

        glm::vec3 wi = glm::refract(-wo, h_world, etaI / etaT);

        if ( glm::dot(wi, wi) < EPSILON || isnan(wi.x) || isnan(wi.y) || isnan(wi.z) || glm::dot(wi, n_facing) >= 0.0f ) {
            path.sample_dir = glm::vec3(0.0f);
            return;
        }

        path.sample_dir = glm::normalize(wi);

    }

    __device__ float PDF_GGX( glm::vec3 wo, glm::vec3 wi, glm::vec3 n, float roughness) {
        glm::vec3 h = glm::normalize(wo + wi);

        roughness = glm::clamp(roughness, 0.0001f, 1.0f);
        float alpha = roughness * roughness;

        float p_h = D_TrowbridgeReitz(h, n, alpha) * glm::dot(n, h);

        float p_wi = p_h / glm::max( (4 * glm::abs(glm::dot(wo, h))), EPSILON);

        return p_wi;
    }  

    __device__ float PDF_GGX_Transmission( glm::vec3 wo, glm::vec3 wi, glm::vec3 n, float roughness, float ior) {
        float cosThetaI = glm::dot(wo, n);
        bool enter = cosThetaI > 0.f;

        float etaI = enter ? 1.0f : ior;
        float etaT = enter ? ior : 1.0f;

        glm::vec3 h_t = transmissionHalfVector(wo, wi, etaI, etaT);

        roughness = glm::clamp(roughness, 0.0001f, 1.0f);
        float alpha = roughness * roughness;

        float denom = etaT * glm::dot(wi, h_t) + etaI * glm::dot(wo, h_t);

        float p_h = D_TrowbridgeReitz(h_t, n, alpha) * glm::abs(glm::dot(n, h_t));

        float p_wi = p_h * (etaT * etaT * glm::abs(glm::dot(wi, h_t))) / glm::max(denom * denom, EPSILON);

        return p_wi;
    }  
    
    __device__ float PDF(glm::vec3 wo, glm::vec3 wi, glm::vec3 n, float roughness, float metallic, glm::vec3 color, float ior, float transmission) {
        ior = glm::max(ior, 1.0001f);

        glm::vec3 n_facing = glm::dot(wo, n) < 0.0f ? -n : n;

        float pdfDiffuse  = max(0.0f, glm::dot(wi, n_facing)) * INV_PI;
        float pdfSpecular = PDF_GGX(wo, wi, n_facing, roughness);
        float pdfRefract = PDF_GGX_Transmission(wo, wi, n, roughness, ior);

        bool refract = (glm::dot(wi, n) * glm::dot(wo, n)) < 0.0f;

        bool enter = glm::dot(wo, n) > 0.0f;
        float etaI = enter ? 1.0f : ior;
        float etaT = enter ? ior : 1.0f;

        float cos_macro = glm::abs(glm::dot(wo, n));

        glm::vec3 F_approx = Fresnel(cos_macro, color, metallic, etaI, etaT);
        float specularRatio = glm::clamp((F_approx.r + F_approx.g + F_approx.b) / 3.0f, 0.01f, 0.99f);

        float F_macro = FresnelDielectric(cos_macro, etaI, etaT);

        float probRefract = transmission * (1.0f - metallic) * (1.0f - F_macro);
        float probDiffuse = (1.0f - probRefract) * (1.0f - specularRatio) * (1.0f - metallic);
        float probSpecular = 1.0f - probRefract - probDiffuse;

        float pdf;
        if (refract) {
            pdf = probRefract * pdfRefract;
        }
        else {
            pdf = probDiffuse * pdfDiffuse + probSpecular * pdfSpecular;
        }

        return glm::max(pdf, PDF_CLAMP);
    }

    
    __device__ glm::vec3 shadePathCookTorrance(
        PathSegment &path,
        glm::vec3 albedo,
        glm::vec3 normal,
        glm::vec3 wi,
        float roughness,
        float metallic,
        float ior,
        float transmission
    )
    {
        glm::vec3 wo = -path.ray.direction;

        ior = glm::max(ior, 1.0001f);

        bool refract = (glm::dot(wi, normal) * glm::dot(wo, normal)) < 0.0f;

        glm::vec3 brdf;
        if (refract) {
            brdf = BTDF(wo, normal, wi, albedo, roughness, metallic, ior, transmission);
        }
        else {
            brdf = BRDF(wo, normal, wi, albedo, roughness, metallic, ior, transmission);
        }

        #if MICROFACET_REMOVE_FIREFLIES
            brdf = glm::clamp(brdf, glm::vec3(0.0), glm::vec3(1.0));
        #endif // REMOVE_FIREFLIES

        float absdot = glm::abs(glm::dot(wi, normal));

        // MICROFACET PDF CLAMP, THIS IS NECESSARY TO REMOVE FIREFLIES
        return brdf * absdot;
    }   

}