#pragma once 

#include <cuda.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>

#include "config.h"
#include "sceneStructs.h"
#include "texture.h"

__device__ glm::vec2 apply_texture_transform(glm::vec2 uv, const TextureTransform& transform);

__device__ glm::vec3 get_albedo(const Material& material, glm::vec2 uv, const TextureData* textures);


__device__ glm::vec3 get_normal(const Material& material, const TextureData* textures, const ShadeableIntersection& intersection, glm::vec3* out_normal_map);


__device__ glm::vec2 get_metallic_roughness(const Material& material, glm::vec2 uv, const TextureData* textures);

__device__ glm::vec3 get_emission(const Material &material, glm::vec2 uv, const TextureData *textures);
