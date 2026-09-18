#include "scene.h"

#include "utilities.h"

#include <cuda.h>

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "texture.h"

#include <fstream>
#include <iostream>
#include <string>
#include <fmt/format.h>
#include <unordered_map>

using namespace std;

#include "scene/gltf_material_utils.h"

#include "tinygltf/tiny_gltf.h"
#include "tinyexr/tinyexr.h"

#include <numeric>
#include <functional>

static void parse_texture_transform(const tinygltf::ExtensionMap& extensions, TextureTransform& out)
{
    auto it = extensions.find("KHR_texture_transform");
    if (it == extensions.end()) {
        return;
    }

    const auto& ext = it->second;

    if (ext.Has("offset")) {
        const auto& o = ext.Get("offset").Get<tinygltf::Value::Array>();
        out.offset = glm::vec2{
            float(o[0].Get<double>()),
            float(o[1].Get<double>())
        };
    }

    if (ext.Has("scale")) {
        const auto& s = ext.Get("scale").Get<tinygltf::Value::Array>();
        out.scale = glm::vec2{
            float(s[0].Get<double>()),
            float(s[1].Get<double>())
        };
    }

    if (ext.Has("rotation")) {
        out.rotation = float(ext.Get("rotation").Get<double>());
    }
}


void Scene::loadFromGLTF(const std::string& gltfName, std::string exr_path) {
    fmt::println("Loading {} as .glb file", gltfName);

    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    bool ret = loader.LoadBinaryFromFile(&model, &err, &warn, gltfName);
    // bool ret = loader.LoadBinaryFromFile(&model, &err, &warn, filename); // for binary glTF(.glb)

    if (!warn.empty()) {
        printf("Warn: %s\n", warn.c_str());
    }

    if (!err.empty()) {
        printf("Err: %s\n", err.c_str());
    }

    if (!ret) {
        printf("Failed to parse glTF: %s\n", gltfName.c_str());
        exit(1);
    }

    Material defaultMat{};
    defaultMat.color = glm::vec3(1.0, 0.0, 0.0f); // Straight Red
    defaultMat.material_type = MaterialType::Microfacet;
    materials.push_back(defaultMat);
    material_names.push_back("Default Material");

    for (auto& tex : model.textures) {
        int img_index = tex.source;
        const tinygltf::Image& img = model.images[img_index];

        int width = img.width;
        int height = img.height;
        int components = img.component;

        if (components != 4) {
            fmt::println("No support yet for {} component images", components);
            exit(1);
        }

        if (img.bits != 8 && img.bits != 16) {
            fmt::println("No support yet for {} bit images", img.bits);
            exit(1);
        }

        size_t texel_count = static_cast<size_t>(width) * static_cast<size_t>(height);
        size_t expected_bytes = texel_count * 4 * (img.bits / 8);

        if (img.image.size() < expected_bytes) {
            fmt::println("Image {} has {} bytes, expected {}", img_index, img.image.size(), expected_bytes);
            exit(1);
        }

        fmt::println("Loading {} x {} texture of {} bytes ({} bpc)", img.width, img.height, img.image.size(), img.bits);

        std::vector<glm::vec4> data(texel_count);

        if (img.bits == 16) {
            const uint16_t* src = reinterpret_cast<const uint16_t*>(img.image.data());

            for (size_t i = 0; i < texel_count; i++) {
                data[i] = glm::vec4(
                    ((float)src[4 * i]) / 65535.0f,
                    ((float)src[4 * i + 1]) / 65535.0f,
                    ((float)src[4 * i + 2]) / 65535.0f,
                    ((float)src[4 * i + 3]) / 65535.0f
                );
            }
        }
        else {
            const uint8_t* src = img.image.data();

            for (size_t i = 0; i < texel_count; i++) {
                data[i] = glm::vec4(
                    ((float)src[4 * i]) / 255.0f,
                    ((float)src[4 * i + 1]) / 255.0f,
                    ((float)src[4 * i + 2]) / 255.0f,
                    ((float)src[4 * i + 3]) / 255.0f
                );
            }
        }

        TextureHandler::get().load_texture(data, width, height);
    }

    int material_base_offset = materials.size();
    for (auto& mat : model.materials) {
        Material newMaterial{};
        if (mat.pbrMetallicRoughness.baseColorFactor.size() >= 3) {
            newMaterial.color = glm::vec3(
                mat.pbrMetallicRoughness.baseColorFactor[0],
                mat.pbrMetallicRoughness.baseColorFactor[1],
                mat.pbrMetallicRoughness.baseColorFactor[2]
            );
        }
        newMaterial.metallic = mat.pbrMetallicRoughness.metallicFactor;
        newMaterial.roughness = mat.pbrMetallicRoughness.roughnessFactor;
        newMaterial.metallic_roughness_tex = mat.pbrMetallicRoughness.metallicRoughnessTexture.index;

        float emissive_strength = 1.0f;
        if (mat.extensions.find("KHR_materials_emissive_strength") != mat.extensions.end()) {
            const auto& ext = mat.extensions.at("KHR_materials_emissive_strength");

            if (ext.Has("emissiveStrength")) {
                emissive_strength = static_cast<float>(ext.Get("emissiveStrength").GetNumberAsDouble());
            }
        }

        newMaterial.emission.emission_strength = emissive_strength;
        newMaterial.emission.emission_color = glm::vec3(mat.emissiveFactor[0], mat.emissiveFactor[1], mat.emissiveFactor[2]);
        newMaterial.emission.emissive_tex = mat.emissiveTexture.index;
        parse_texture_transform(mat.emissiveTexture.extensions, newMaterial.emission.emissive_tex_transform);

        float ior = 1.55f;
        if (mat.extensions.find("KHR_materials_ior") != mat.extensions.end()) {
            const auto& ext = mat.extensions.at("KHR_materials_ior");

            if (ext.Has("ior")) {
                ior = static_cast<float>(ext.Get("ior").GetNumberAsDouble());
            }
        }

        newMaterial.indexOfRefraction = ior;

        float transmission = 0.0f;
        if (mat.extensions.find("KHR_materials_transmission") != mat.extensions.end()) {
            const auto& ext = mat.extensions.at("KHR_materials_transmission");

            if (ext.Has("transmissionFactor")) {
                transmission = static_cast<float>(ext.Get("transmissionFactor").GetNumberAsDouble());
            }
        }

        newMaterial.transmission = transmission;

        #if UBER_SHADER
            newMaterial.material_type = MaterialType::Microfacet;
        #else
            if (emissive_strength * glm::length(newMaterial.emission.emission_color) > EPSILON) {
                newMaterial.material_type = MaterialType::Emissive;
                newMaterial.color = glm::vec3(mat.emissiveFactor[0], mat.emissiveFactor[1], mat.emissiveFactor[2]);
            }
            else if (isGlass(mat) && newMaterial.metallic < 0.01f && newMaterial.roughness < 0.01f) {
                newMaterial.material_type = MaterialType::Glass;
            }
            else if (newMaterial.metallic_roughness_tex >= 0) {
                newMaterial.material_type = MaterialType::Microfacet;
            }
            else if (newMaterial.metallic < 0.01f && newMaterial.roughness > 0.99f) {
                newMaterial.material_type = MaterialType::Microfacet;
            }
            else {
                newMaterial.material_type = MaterialType::Microfacet;
            }
        #endif

        auto& albedo_tex_info = mat.pbrMetallicRoughness.baseColorTexture;        
        newMaterial.albedo_tex = albedo_tex_info.index;
        parse_texture_transform(albedo_tex_info.extensions, newMaterial.albedo_tex_transform);

        newMaterial.normal_tex = mat.normalTexture.index;
        parse_texture_transform(mat.normalTexture.extensions, newMaterial.normal_tex_transform);

        parse_texture_transform(mat.pbrMetallicRoughness.metallicRoughnessTexture.extensions, newMaterial.metallic_roughness_tex_transform);


        materials.push_back(newMaterial);
        material_names.push_back(mat.name);
        // fmt::println("New Material {} with RGB {} of type {} at index {} and has texture {}", mat.name, glm::to_string(newMaterial.color), (int)newMaterial.material_type, materials.size() - 1, newMaterial.albedo_tex >= 0);
    }

    int camera_node = -1;
    glm::mat4 camera_transform;

    std::function<void(int, glm::mat4)> processNode;
    processNode = [&](int node_idx, glm::mat4 parent_transform)
    {
        const auto& node = model.nodes[node_idx];
        glm::mat4 node_transform = glm::mat4(1.0f);

        if (!node.matrix.empty()) {
            node_transform = glm::make_mat4(node.matrix.data());
        }
        else {
            if (!node.translation.empty()) {
                node_transform = glm::translate(
                    node_transform, 
                    glm::vec3(
                        node.translation[0], 
                        node.translation[1], 
                        node.translation[2]
                    )
                );
            }

            if (!node.rotation.empty()) {
                glm::quat q(
                    node.rotation[3], 
                    node.rotation[0], 
                    node.rotation[1], 
                    node.rotation[2]
                );
                node_transform *= glm::mat4_cast(q);
            }

            if (!node.scale.empty()) {
                node_transform = glm::scale(
                    node_transform,
                    glm::vec3(
                        node.scale[0],
                        node.scale[1],
                        node.scale[2]
                    )
                );
            }
        }

        glm::mat4 global_transform = parent_transform * node_transform;

        if (node.mesh >= 0) {
            const auto& mesh = model.meshes[node.mesh];
            for (size_t prim_idx = 0; prim_idx < mesh.primitives.size(); prim_idx++) {
                const auto& prim = mesh.primitives[prim_idx];
                if (prim.mode != TINYGLTF_MODE_TRIANGLES) {
                    fmt::println("WARNING: Mesh {} attempted to create a non-triangle mode primitive", mesh.name);
                    continue;
                }

                Geom new_geom{};

                new_geom.type = GeomType::MESH;

                auto it = prim.attributes.find("POSITION");
                if (it == prim.attributes.end()) { 
                    continue;
                }
                const auto& pos_accessor = model.accessors[it->second];

                assert(pos_accessor.type == TINYGLTF_TYPE_VEC3);
                assert(pos_accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT);

                const auto& pos_buffer_view = model.bufferViews[pos_accessor.bufferView];
                const auto& buffer = model.buffers[pos_buffer_view.buffer];

                std::vector<glm::vec3> vertices(pos_accessor.count);

                size_t stride = pos_buffer_view.byteStride
                    ? pos_buffer_view.byteStride
                    : sizeof(float) * 3;

                const uint8_t* base =
                    buffer.data.data() +
                    pos_buffer_view.byteOffset +
                    pos_accessor.byteOffset;

                for (size_t i = 0; i < pos_accessor.count; i++) {
                    const float* p = reinterpret_cast<const float*>(base + i * stride);
                    vertices[i] = { p[0], p[1], p[2] };
                }

                std::vector<glm::vec3> normals {};
                auto it_norm = prim.attributes.find("NORMAL");

                if (it_norm != prim.attributes.end()) {
                    const auto& norm_accessor = model.accessors[it_norm->second];
                    const auto& norm_buffer_view = model.bufferViews[norm_accessor.bufferView];
                    const auto& norm_buffer = model.buffers[norm_buffer_view.buffer];

                    normals.resize(norm_accessor.count);

                    size_t norm_stride = norm_buffer_view.byteStride 
                        ? norm_buffer_view.byteStride 
                        : sizeof(float) * 3;

                    const uint8_t* norm_base = 
                        norm_buffer.data.data() + 
                        norm_buffer_view.byteOffset + 
                        norm_accessor.byteOffset;

                    for (size_t i = 0; i < norm_accessor.count; i++) {
                        const float* n = reinterpret_cast<const float*>(norm_base + i * norm_stride);
                        normals[i] = glm::normalize(glm::vec3{ n[0], n[1], n[2] });
                    }
                }

                
                std::vector<glm::vec2> uvs {};
                auto it_uv = prim.attributes.find("TEXCOORD_0");

                if (it_uv != prim.attributes.end()) {
                    const auto& uv_accessor = model.accessors[it_uv->second];
                    const auto& uv_buffer_view = model.bufferViews[uv_accessor.bufferView];
                    const auto& uv_buffer = model.buffers[uv_buffer_view.buffer];

                    assert(uv_accessor.type == TINYGLTF_TYPE_VEC2);
                    assert(uv_accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT);

                    uvs.resize(uv_accessor.count);

                    size_t uv_stride = uv_buffer_view.byteStride 
                        ? uv_buffer_view.byteStride 
                        : sizeof(float) * 2;

                    const uint8_t* uv_base = 
                        uv_buffer.data.data() + 
                        uv_buffer_view.byteOffset + 
                        uv_accessor.byteOffset;

                    for (size_t i = 0; i < uv_accessor.count; i++) {
                        const float* uv = reinterpret_cast<const float*>(uv_base + i * uv_stride);
                        uvs[i] = glm::vec2{ uv[0], uv[1] };
                    }
                }


                std::vector<int> indices;
                if (prim.indices >= 0) {
                    const auto& idx_accessor = model.accessors[prim.indices];
                    const auto& idx_buffer_view = model.bufferViews[idx_accessor.bufferView];
                    const auto& idx_buffer = model.buffers[idx_buffer_view.buffer];
                    const void* idx_data = &idx_buffer.data[idx_buffer_view.byteOffset + idx_accessor.byteOffset];

                    indices.resize(idx_accessor.count);
                    if (idx_accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                        const uint16_t* buf = static_cast<const uint16_t*>(idx_data);
                        for (size_t i = 0; i < idx_accessor.count; ++i) {
                            indices[i] = buf[i];
                        }
                    } 
                    else if (idx_accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
                        const uint32_t* buf = static_cast<const uint32_t*>(idx_data);
                        for (size_t i = 0; i < idx_accessor.count; ++i) {
                            indices[i] = buf[i];
                        }
                    }
                    else if (idx_accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
                        const uint8_t* buf = static_cast<const uint8_t*>(idx_data);
                        for (size_t i = 0; i < idx_accessor.count; ++i) { 
                            indices[i] = buf[i];
                        }
                    }
                } 
                else {
                    indices.resize(vertices.size());
                    std::iota(indices.begin(), indices.end(), 0);
                }

                new_geom.mesh.make_mesh_host(vertices, indices, normals, indices, uvs, indices);
                new_geom.mesh.label = fmt::format("{}#{}#{}", node.mesh, prim_idx, mesh.name);
                new_geom.transform = global_transform;
                new_geom.inverseTransform = glm::inverse(global_transform);
                new_geom.invTranspose = glm::inverseTranspose(global_transform);

                new_geom.materialid = (prim.material >= 0) ? prim.material + material_base_offset : 0; // default material id
                geoms.push_back(new_geom);
            }
        }

        if (node.camera >= 0) {
            if (camera_node != -1) {
                fmt::println("Multiple cameras defined in GLTF, exiting");
                exit(1);
            }

            camera_node = node_idx;
            camera_transform = global_transform;
        }

        for (auto child : node.children) {
            processNode(child, global_transform);
        }
    };

    int sceneIndex = (model.defaultScene >= 0) ? model.defaultScene : 0;
    for (size_t i = 0; i < model.scenes[sceneIndex].nodes.size(); i++) {
        processNode(model.scenes[sceneIndex].nodes[i], glm::mat4(1.0f));
    }

    if (camera_node < 0) {
        fmt::println("No camera found in glTF");
        exit(1);
    }

    tinygltf::Camera& gltf_camera = model.cameras[model.nodes[camera_node].camera];
    fmt::println("Found camera {} at node {}", gltf_camera.name, camera_node);

    float aspect = static_cast<float>(gltf_camera.perspective.aspectRatio);

    Camera& camera = state.camera;
    RenderState& state = this->state;

    if (aspect >= 1.0) {
        camera.resolution.y = 1000;
        camera.resolution.x = static_cast<int>(aspect * camera.resolution.y);
    } else {
        camera.resolution.x = 1000;
        camera.resolution.y = static_cast<int>(camera.resolution.x / aspect);
    }

    float fovy = glm::degrees(gltf_camera.perspective.yfov);
    state.iterations = 5000; // this should be customized
    state.traceDepth = 8; // this should be customized
    state.imageName = model.scenes[sceneIndex].name.empty() ? "GLTF_DEFAULT_NAME" : model.scenes[sceneIndex].name;

    glm::vec3 eye = glm::vec3(camera_transform[3]);

    camera.position = eye;
    
    camera.view  = -glm::normalize(glm::vec3(camera_transform[2]));
    camera.right = glm::normalize(glm::vec3(camera_transform[0])); 
    camera.up    = glm::normalize(glm::vec3(camera_transform[1]));

    camera.lookAt = camera.position + camera.view;

    // //calculate fov based on resolution
    float yscaled = tan(0.5f * fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (2.0f * atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    // //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());

    if (!exr_path.empty()) {
        float* exr;
        const char* exr_err = NULL;

        int exr_ret = LoadEXR(&exr, &exr_width, &exr_height, exr_path.c_str(), &exr_err);

        if (exr_ret != TINYEXR_SUCCESS) {
            if (exr_err) {
                fprintf(stderr, "ERR : %s\n", exr_err);
                FreeEXRErrorMessage(exr_err);
            }
            exit(1);
        } else {
            fmt::println("Exr loading success: {} x {}", exr_width, exr_height);

            exr_data.resize(exr_width * exr_height);
            memcpy(exr_data.data(), exr, 4 * exr_width * exr_height * sizeof(float));

            free(exr);
        }
    }

    precompute_emissive_mesh_area();

    if (!exr_path.empty()) {
        precompute_hdri_emission();
    }
}
