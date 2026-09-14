#pragma once

#include <cmath>

#include "tinygltf/tiny_gltf.h"

#if !UBER_SHADER

inline bool isGlass(const tinygltf::Material& mat) {
    // 1. Explicit Transmission (Modern standard)
    if (mat.extensions.count("KHR_materials_transmission")) return true;

    // 2. Volume (If it has internal absorption/thickness, it's glass)
    if (mat.extensions.count("KHR_materials_volume")) return true;

    // 3. IOR check (If IOR is set and not 1.0, it's likely meant to be refractive)
    if (mat.extensions.count("KHR_materials_ior")) {
        auto it = mat.extensions.find("KHR_materials_ior");
        if (it->second.Has("ior")) {
            double ior = it->second.Get("ior").GetNumberAsDouble();
            if (std::abs(ior - 1.0) > 0.01) return true;
        }
    }

    // 4. Check Alpha in PBR block
    if (mat.alphaMode == "BLEND" || mat.alphaMode == "MASK") {
         if (mat.pbrMetallicRoughness.baseColorFactor[3] < 0.99f) return true;
    }

    // 5. Specular Glossiness Workflow Fallback
    if (mat.extensions.count("KHR_materials_pbrSpecularGlossiness")) {
        auto it = mat.extensions.find("KHR_materials_pbrSpecularGlossiness");
        if (it->second.Has("diffuseFactor")) {
            auto factor = it->second.Get("diffuseFactor");
            // Check the 4th element (alpha) of the diffuse array
            if (factor.IsArray() && factor.ArrayLen() == 4) {
                if (factor.Get(3).GetNumberAsDouble() < 0.99) return true;
            }
        }
    }

    return false;
}

#endif