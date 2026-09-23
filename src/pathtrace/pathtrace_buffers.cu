#include "pathtrace/pathtrace_buffers.h"

#include "pathtrace/pathtrace_state.h"
#include "pathtrace/dev_options.h"

#include "scene.h"
#include "sceneStructs.h"
#include "utilities.h"
#include "common.h"
#include "myoptix.h"
#include "texture.h"
#include "pathtrace.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <glm/glm.hpp>

#include <fmt/core.h>

void pathtraceInit(Scene* scene)
{
    PathTraceState& pt_state = PathTraceState::Get();
    pt_state.hst_scene = scene;

    const Camera& cam = pt_state.hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&pt_state.dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(pt_state.dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&pt_state.dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&pt_state.dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(pt_state.dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&pt_state.dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(pt_state.dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&pt_state.dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(pt_state.dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&pt_state.dev_direct_light_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(pt_state.dev_direct_light_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&pt_state.dev_environment_map_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(pt_state.dev_environment_map_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&pt_state.dev_morton_codes, pixelcount * sizeof(uint32_t));

    cudaMalloc(&pt_state.dev_path_indices_A, pixelcount * sizeof(int));
    cudaMalloc(&pt_state.dev_path_indices_B, pixelcount * sizeof(int));
    cudaMalloc(&pt_state.dev_num_active_paths, sizeof(int));

    cudaMalloc( reinterpret_cast<void**>( &pt_state.d_optix_paramters ), sizeof( Params ) );

    cudaMalloc( &pt_state.dev_material_ids, sizeof(int) * scene->geoms.size());
    std::vector<int> material_ids;
    for (Geom& geom : scene->geoms) {
        material_ids.push_back(geom.materialid);
    }
    cudaMemcpy(pt_state.dev_material_ids, material_ids.data(), material_ids.size() * sizeof(int), cudaMemcpyHostToDevice);

    std::vector<OptixAlphaMaterial> alpha_materials;
    for (const Material& mat : scene->materials) {
        OptixAlphaMaterial alpha_mat = {};
        alpha_mat.alpha_mode = mat.alpha_mode;
        alpha_mat.alpha = mat.alpha;
        alpha_mat.alpha_cutoff = mat.alpha_cutoff;
        alpha_mat.albedo_tex = mat.albedo_tex >= 0 ? TextureHandler::get().host_textures[mat.albedo_tex].tex : 0;
        alpha_mat.tex_offset = make_float2(mat.albedo_tex_transform.offset.x, mat.albedo_tex_transform.offset.y);
        alpha_mat.tex_scale = make_float2(mat.albedo_tex_transform.scale.x, mat.albedo_tex_transform.scale.y);
        alpha_mat.tex_rotation = mat.albedo_tex_transform.rotation;
        alpha_materials.push_back(alpha_mat);
    }
    cudaMalloc( &pt_state.dev_alpha_materials, sizeof(OptixAlphaMaterial) * alpha_materials.size());
    cudaMemcpy(pt_state.dev_alpha_materials, alpha_materials.data(), alpha_materials.size() * sizeof(OptixAlphaMaterial), cudaMemcpyHostToDevice);

    std::vector<glm::vec3*> vertex_buffer_locs;
    std::vector<Triangle*> triangle_buffer_locs;
    std::vector<glm::vec3*> normal_buffer_locs;
    std::vector<glm::vec2*> uv_buffer_locs;

    for (const Geom& geo : scene->geoms) {
        if (geo.type == GeomType::MESH) {
            vertex_buffer_locs.push_back(geo.mesh.d_verts);
            triangle_buffer_locs.push_back(geo.mesh.d_triangles);
            normal_buffer_locs.push_back(geo.mesh.has_normal_buffers ? geo.mesh.d_normals : nullptr);
            uv_buffer_locs.push_back(geo.mesh.has_uvs ? geo.mesh.d_uvs : nullptr);
        }
    }

    cudaMalloc( &pt_state.dev_vertex_buffer_locs, sizeof(glm::vec3*) * vertex_buffer_locs.size() );
    cudaMalloc( &pt_state.dev_triangle_buffer_locs, sizeof(Triangle*) * triangle_buffer_locs.size() );
    cudaMalloc( &pt_state.dev_normal_buffer_locs, sizeof(glm::vec3*) * normal_buffer_locs.size() );
    cudaMalloc( &pt_state.dev_uv_buffer_locs, sizeof(glm::vec2*) * uv_buffer_locs.size() );

    cudaMemcpy( pt_state.dev_vertex_buffer_locs, vertex_buffer_locs.data(), sizeof(glm::vec3*) * vertex_buffer_locs.size(), cudaMemcpyHostToDevice);
    cudaMemcpy( pt_state.dev_triangle_buffer_locs, triangle_buffer_locs.data(), sizeof(Triangle*) * triangle_buffer_locs.size(), cudaMemcpyHostToDevice);
    cudaMemcpy( pt_state.dev_normal_buffer_locs, normal_buffer_locs.data(), sizeof(glm::vec3*) * normal_buffer_locs.size(), cudaMemcpyHostToDevice);
    cudaMemcpy( pt_state.dev_uv_buffer_locs, uv_buffer_locs.data(), sizeof(glm::vec2*) * uv_buffer_locs.size(), cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(DEV_OPTIONS, PathTracerOptions::Get(), sizeof(PathTracerOptions));

    if (!scene->exr_data.empty()) {
        // exr loading on GPU
        cudaChannelFormatDesc exr_channel_desc = cudaCreateChannelDesc<float4>();
        cudaMallocArray(&pt_state.dev_exr_array, &exr_channel_desc, scene->exr_width, scene->exr_height);
        cudaMemcpyToArray(pt_state.dev_exr_array, 0, 0, scene->exr_data.data(), scene->exr_width * scene->exr_height * sizeof(glm::vec4), cudaMemcpyHostToDevice);

        cudaResourceDesc res_desc = {};
        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = pt_state.dev_exr_array;

        cudaTextureDesc tex_desc = {};
        tex_desc.addressMode[0] = cudaAddressModeWrap;
        tex_desc.addressMode[1] = cudaAddressModeWrap;
        tex_desc.filterMode = cudaFilterModeLinear;
        tex_desc.readMode = cudaReadModeElementType;
        tex_desc.normalizedCoords = 1;

        cudaCreateTextureObject(&pt_state.exr_texture, &res_desc, &tex_desc, nullptr);
    }

    cudaMalloc( &pt_state.dev_emissive_geoms, scene->emissive_geoms.size() * sizeof(int));
    cudaMemcpy( pt_state.dev_emissive_geoms, scene->emissive_geoms.data(), scene->emissive_geoms.size() * sizeof(int), cudaMemcpyHostToDevice );

    cudaMalloc( &pt_state.dev_emissive_geom_area_prefix, scene->emissive_geom_area_prefix.size() * sizeof(float));
    cudaMemcpy( pt_state.dev_emissive_geom_area_prefix, scene->emissive_geom_area_prefix.data(), scene->emissive_geom_area_prefix.size() * sizeof(float), cudaMemcpyHostToDevice );

    if (!pt_state.hst_scene->exr_data.empty()) {
        cudaMalloc( &pt_state.dev_hdri_conditional_cdfs, scene->hdri_conditional_cdfs.size() * sizeof(float) );
        cudaMemcpy( pt_state.dev_hdri_conditional_cdfs, scene->hdri_conditional_cdfs.data(), scene->hdri_conditional_cdfs.size() * sizeof(float), cudaMemcpyHostToDevice );

        cudaMalloc( &pt_state.dev_hdri_marginal_cdf, scene->hdri_marginal_cdf.size() * sizeof(float) );
        cudaMemcpy( pt_state.dev_hdri_marginal_cdf, scene->hdri_marginal_cdf.data(), scene->hdri_marginal_cdf.size() * sizeof(float), cudaMemcpyHostToDevice );
    }

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    PathTraceState& pt_state = PathTraceState::Get();
    cudaFree(pt_state.dev_image);
    cudaFree(pt_state.dev_paths);
    cudaFree(pt_state.dev_geoms);
    cudaFree(pt_state.dev_materials);
    cudaFree(pt_state.dev_intersections);
    cudaFree(pt_state.dev_direct_light_intersections);
    cudaFree(pt_state.dev_environment_map_intersections);

    cudaFree(pt_state.dev_morton_codes);
    cudaFree(pt_state.dev_path_indices_A);
    cudaFree(pt_state.dev_path_indices_B);
    cudaFree(pt_state.dev_num_active_paths);
    cudaFree(pt_state.dev_material_ids);
    cudaFree(pt_state.dev_alpha_materials);

    cudaFree(pt_state.dev_vertex_buffer_locs);
    cudaFree(pt_state.dev_triangle_buffer_locs);
    cudaFree(pt_state.dev_normal_buffer_locs);
    cudaFree(pt_state.dev_uv_buffer_locs);

    cudaFree(pt_state.dev_emissive_geoms);
    cudaFree(pt_state.dev_emissive_geom_area_prefix);
    cudaFree(pt_state.dev_hdri_marginal_cdf);
    cudaFree(pt_state.dev_hdri_conditional_cdfs);
    
    cudaFree(reinterpret_cast<void*>(pt_state.d_optix_paramters));

    // free exr
    cudaDestroyTextureObject(pt_state.exr_texture);
    cudaFreeArray(pt_state.dev_exr_array);

    checkCUDAError("pathtraceFree");
}

// Retrieve image from GPU
void copyImageToHost()
{
    PathTraceState& pt_state = PathTraceState::Get();
    const Camera& cam = pt_state.hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMemcpy(pt_state.hst_scene->state.image.data(), pt_state.dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("copyImageToHost");
}
