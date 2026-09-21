#include "material_queries.h"

__device__ glm::vec2 apply_texture_transform(glm::vec2 uv, const TextureTransform& transform) {
    uv *= transform.scale;

    if(transform.rotation) {
        float c = cosf(transform.rotation);
        float s = sinf(transform.rotation);

        uv = glm::vec2 (
            c * uv.x - s * uv.y,
            s * uv.x + c * uv.y
        );
    }

    return uv + transform.offset;
}

__device__ glm::vec3 get_albedo(const Material& material, glm::vec2 uv, const TextureData* textures) {
    if(material.albedo_tex >= 0) {
        uv = apply_texture_transform(uv, material.albedo_tex_transform);

        float4 tex = tex2D<float4>(textures[material.albedo_tex].tex, uv.x, uv.y);
        return glm::pow(glm::vec3(tex.x, tex.y, tex.z), glm::vec3(2.2f));
    }
    else {
        return material.color;
    }
}


__device__ glm::vec3 get_normal(const Material& material, const TextureData* textures, const ShadeableIntersection& intersection, glm::vec3* out_normal_map) {
    if (material.normal_tex >= 0) {
        glm::vec2 uv = apply_texture_transform(intersection.uvs, material.normal_tex_transform);

        float4 tex = tex2D<float4>(textures[material.normal_tex].tex, uv.x, uv.y);
        glm::vec3 local_normal = glm::vec3(tex.x, tex.y, tex.z);

        *out_normal_map = local_normal;

        local_normal.x = local_normal.r * 2.0f - 1.0f;
        local_normal.y = local_normal.g * 2.0f - 1.0f;
        local_normal.z = local_normal.b * 2.0f - 1.0f;

        const glm::vec3 N = intersection.surfaceNormal;

        // fallback for bad normal
        float local_len2 = glm::dot(local_normal, local_normal);
        if (!(local_len2 > 1e-12f)) {
            return N;
        }

        // forcing the tangent to be orthogonal

        // tangent zero guard 
        glm::vec3 T = intersection.surfaceTangent - glm::dot(intersection.surfaceTangent, N) * N;
        float t_len2 = glm::dot(T, T);
        if (!(t_len2 > 1e-12f)) {
            return N;
        }
        T *= glm::inversesqrt(t_len2);

        glm::vec3 bitangent = glm::cross(N, T);

        glm::mat3 TBN = glm::mat3(T, bitangent, N);

        // bad local normal guard
        glm::vec3 mapped = TBN * local_normal;
        float mapped_len2 = glm::dot(mapped, mapped);
        if (!(mapped_len2 > 1e-12f)) {
            return N;
        }
        mapped *= glm::inversesqrt(mapped_len2);

        // another normal guard (mapped normal not same dir as world normal)
        if (!(glm::dot(mapped, N) > 1e-4f)) {
            return N;
        }

        return mapped;
    }
    else {
        *out_normal_map = glm::vec3(0.5f, 0.5f, 1.0f);
        return intersection.surfaceNormal;
    }
}


__device__ glm::vec2 get_metallic_roughness(const Material& material, glm::vec2 uv, const TextureData* textures) {
    float roughness = material.roughness;
    float metallic = material.metallic;

    if (material.metallic_roughness_tex >= 0) {
        uv = apply_texture_transform(uv, material.metallic_roughness_tex_transform);

        float4 tex = tex2D<float4>(textures[material.metallic_roughness_tex].tex, uv.x, uv.y);

        roughness *= tex.y;
        metallic *= tex.z;
    }

    return glm::vec2(roughness, metallic);
}

__device__ glm::vec3 get_emission(const Material &material, glm::vec2 uv, const TextureData *textures) {
    glm::vec3 emission = material.emission.emission_color * material.emission.emission_strength;

    if (material.emission.emissive_tex >= 0) {
        uv = apply_texture_transform(uv, material.emission.emissive_tex_transform);

        float4 tex = tex2D<float4>(textures[material.emission.emissive_tex].tex, uv.x, uv.y);

        emission *= glm::pow(glm::vec3(tex.x, tex.y, tex.z), glm::vec3(2.2f));
    }

    return emission;
}

__device__ float get_alpha(const Material& material, glm::vec2 uv, const TextureData* textures) {
    if (material.alpha_mode == ALPHA_MODE_OPAQUE) {
        return 1.0f;
    }

    float alpha = material.alpha;
    if(material.albedo_tex >= 0) {
        uv = apply_texture_transform(uv, material.albedo_tex_transform);

        float4 tex = tex2D<float4>(textures[material.albedo_tex].tex, uv.x, uv.y);
        alpha *= tex.w;
    }

    if (material.alpha_mode == ALPHA_MODE_MASK) {
        return alpha >= material.alpha_cutoff ? 1.0f : 0.0f;
    }

    return alpha;
}