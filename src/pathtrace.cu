#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/system/cuda/execution_policy.h>

#include <map>
#include <new>
#include <utility>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/common.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "stack.h"
#include "cuda_timer.h"

#include <fmt/core.h>

#include "common.h"
#include "myoptix.h"
#include "texture.h"
#include "tonemapping.h"
#include "material_queries.h"
#include "material_debug_render.h"
#include "sample_materials.h"
#include "update_throughput_materials.h"
#include "morton_codes.h"
#include "pathtrace/dev_options.h"
#include "pathtrace/pathtrace_state.h"

#include "kernels/image_kernels.h"
#include "kernels/generate_ray_from_camera.h"
#include "kernels/compute_intersections.h"
#include "kernels/shade_path.h"
#include "kernels/intersection_precompute.h"
#include "kernels/draw_bvh.h"
#include "kernels/sample_direct_light.h"
#include "pathtrace/path_functors.h"

#include "shaders/lambert.h"
#include "shaders/specular.h"
#include "shaders/cook_torrance.h"
#include "shaders/glass.h"  

struct ThrustCachingAllocator
{
    typedef char value_type;

    std::multimap<std::ptrdiff_t, char*> free_blocks;
    std::map<char*, std::ptrdiff_t> allocated_blocks;

    char* allocate(std::ptrdiff_t num_bytes)
    {
        char* ptr = nullptr;
        auto free_block = free_blocks.lower_bound(num_bytes);
        if (free_block != free_blocks.end()) {
            num_bytes = free_block->first;
            ptr = free_block->second;
            free_blocks.erase(free_block);
        }
        else if (cudaMalloc(reinterpret_cast<void**>(&ptr), num_bytes) != cudaSuccess) {
            throw std::bad_alloc();
        }
        allocated_blocks.emplace(ptr, num_bytes);
        return ptr;
    }

    void deallocate(char* ptr, size_t)
    {
        auto it = allocated_blocks.find(ptr);
        free_blocks.emplace(it->second, it->first);
        allocated_blocks.erase(it);
    }
};

static ThrustCachingAllocator thrust_allocator;






void pathtrace(uchar4* pbo, int frame, int iter)
{
    PathTraceState& pt_state = PathTraceState::Get();
    // fmt::println("PATHTRACE: {} vs {}", sizeof(ShadeableIntersection), sizeof(OptixShadeableIntersection));
    // fmt::println("Offset 0: {} vs {}", offsetof(ShadeableIntersection, t), offsetof(OptixShadeableIntersection, t));
    // fmt::println("Offset 1: {} vs {}", offsetof(ShadeableIntersection, surfaceNormal), offsetof(OptixShadeableIntersection, surfaceNormal));
    // fmt::println("Offset 2: {} vs {}", offsetof(ShadeableIntersection, surfaceTangent), offsetof(OptixShadeableIntersection, surfaceTangent));
    // fmt::println("Offset 3: {} vs {}", offsetof(ShadeableIntersection, materialId), offsetof(OptixShadeableIntersection, materialId));
    // fmt::println("Offset 4: {} vs {}", offsetof(ShadeableIntersection, uvs), offsetof(OptixShadeableIntersection, u));

    const int traceDepth = PathTracerOptions::Get()->material_debug_mode ? 1 : pt_state.hst_scene->state.traceDepth;
    const Camera& cam = pt_state.hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = BLOCK_SIZE_1D;

    CudaTimer cudaTimer;

    CUDA_TIMER_RECORD(cudaTimer, "Start, Iter {}", 1);

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, pt_state.dev_paths, pt_state.dev_path_indices_A);
    checkCUDAError("generate camera ray");

    CUDA_TIMER_RECORD(cudaTimer, "Generate Camera Rays, Iter {}", 1);

    int depth = 0;
    int num_paths = pixelcount;

    PathSegment* dev_paths = pt_state.dev_paths;
    int* dev_path_indices = pt_state.dev_path_indices_A;
    int* dev_next_path_indices = pt_state.dev_path_indices_B;

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        if (depth > 0) {
            CUDA_TIMER_RECORD(cudaTimer, "Start, Iter {}", depth+1);
        }

        if (iter == MAX_ITERATIONS) {
            exit(0);
        }

        #if !OPTIX
            cudaMemset(pt_state.dev_intersections, 0, num_paths * sizeof(ShadeableIntersection));
            cudaMemset(pt_state.dev_direct_light_intersections, 0, num_paths * sizeof(ShadeableIntersection));
            cudaMemset(pt_state.dev_environment_map_intersections, 0, num_paths * sizeof(ShadeableIntersection));

            CUDA_TIMER_RECORD(cudaTimer, "Memset, Iter {}", depth+1);
        #endif

        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

        #if RAY_SORTING
            const Geom* d_mesh = nullptr;
            for (int i = 0; i < pt_state.hst_scene->geoms.size(); i++) {
                Geom &geom = pt_state.hst_scene->geoms[i];
                if (geom.type == GeomType::MESH) {
                    d_mesh = pt_state.dev_geoms + i;
                    break;
                }
            }

            if (d_mesh == nullptr) {
                printf("ERROR: No Mesh Detected\n");
                exit(1);
            }

            intersectionPrecompute<<<numblocksPathSegmentTracing, blockSize1d>>> (
                num_paths,
                dev_paths,
                dev_path_indices,
                d_mesh,
                pt_state.dev_morton_codes
            );

            CUDA_TIMER_RECORD(cudaTimer, "Morton Precompute, Iter {}", depth+1);

            thrust::sort_by_key(thrust::cuda::par(thrust_allocator), dPtr(pt_state.dev_morton_codes), dPtr(pt_state.dev_morton_codes + num_paths), dPtr(dev_path_indices));

            CUDA_TIMER_RECORD(cudaTimer, "Sort Mesh Hits Morton, Iter {}", depth+1);
        #endif

        if (PathTracerOptions::Get()->direct_light_sampling || PathTracerOptions::Get()->environment_map_importance_sampling) {
            sampleDirectLight<<<numblocksPathSegmentTracing, blockSize1d>>> (
                iter, 
                num_paths, 
                dev_paths, 
                dev_path_indices,
                depth,
                pt_state.hst_scene->emissive_geoms.size(),
                pt_state.dev_emissive_geoms,
                pt_state.dev_emissive_geom_area_prefix,
                pt_state.dev_geoms,
                pt_state.hst_scene->total_emissive_mesh_area,
                pt_state.dev_hdri_marginal_cdf,
                pt_state.dev_hdri_conditional_cdfs,
                pt_state.hst_scene->exr_width,
                pt_state.hst_scene->exr_height
            );

            CUDA_TIMER_RECORD(cudaTimer, "Sample Lights, Iter {}", depth+1);
        }

        if (PathTracerOptions::Get()->debug_bvh) {
            drawBVH<<<numblocksPathSegmentTracing, blockSize1d>>> (
                depth,
                num_paths,
                dev_paths,
                dev_path_indices,
                pt_state.dev_geoms,
                pt_state.hst_scene->geoms.size(),
                pt_state.dev_intersections
            );

            CUDA_TIMER_RECORD(cudaTimer, "Draw BVH, Iter {}", depth+1);
        }
        else {
            #if !OPTIX
                // tracing
                computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
                    depth,
                    num_paths,
                    dev_paths,
                    dev_path_indices,
                    pt_state.dev_geoms,
                    pt_state.hst_scene->geoms.size(),
                    pt_state.dev_intersections
                );
                checkCUDAError("compute intersections");

                CUDA_TIMER_RECORD(cudaTimer, "Compute Intersections, Iter {}", depth+1);
            #else // OPTIX
                // time for some optix magic
                Params optix_params = {};
                optix_params.handle = pt_state.hst_scene->ias_handle;
                optix_params.path_segments = reinterpret_cast<OptixPathSegment*>(dev_paths);
                optix_params.path_indices = dev_path_indices;
                optix_params.debug_image = reinterpret_cast<float3*>(pt_state.dev_image);
                optix_params.shadeable_intersections = reinterpret_cast<OptixShadeableIntersection*>(pt_state.dev_intersections);
                optix_params.direct_light_intersections = reinterpret_cast<OptixShadeableIntersection*>(pt_state.dev_direct_light_intersections);
                optix_params.environment_map_intersections = reinterpret_cast<OptixShadeableIntersection*>(pt_state.dev_environment_map_intersections);
                optix_params.material_ids = pt_state.dev_material_ids;
                optix_params.alpha_materials = pt_state.dev_alpha_materials;
                optix_params.iteration = iter;
                optix_params.vertex_buffer_locations = (float3**)pt_state.dev_vertex_buffer_locs;
                optix_params.triangle_buffer_locations = (OptixTriangle**)pt_state.dev_triangle_buffer_locs;
                optix_params.normal_buffer_locations = (float3**)pt_state.dev_normal_buffer_locs;
                optix_params.uv_buffer_locations = (float2**)pt_state.dev_uv_buffer_locs;
                optix_params.direct_light_sampling = PathTracerOptions::Get()->direct_light_sampling;
                optix_params.envmap_sampling = PathTracerOptions::Get()->environment_map_importance_sampling;
                cudaMemcpyAsync(reinterpret_cast<void*>(pt_state.d_optix_paramters), &optix_params, sizeof(Params), cudaMemcpyHostToDevice, 0);
                OPTIX_CHECK(
                    optixLaunch(pt_state.hst_scene->optix_pipeline, 0, pt_state.d_optix_paramters, sizeof(Params), &pt_state.hst_scene->optix_sbt, num_paths, 1, 1);
                );
                // fmt::println("OptixTrace Iteration {}", iter);
                // end of optix magic
                CUDA_TIMER_RECORD(cudaTimer, "Optix Compute Intersections, Iter {}", depth+1);
            #endif //OPTIX

            depth++;

            #if MATERIAL_SORTING
                thrust::sort_by_key(
                    thrust::cuda::par(thrust_allocator),
                    pt_state.dev_intersections,
                    pt_state.dev_intersections + num_paths,
                    dev_path_indices,
                    sort_materials()
                );

                CUDA_TIMER_RECORD(cudaTimer, "Material Sorting, Iter {}", depth);
            #endif

            #if STREAM_COMPACTION
                const bool compact_paths = depth < traceDepth;
            #else
                const bool compact_paths = false;
            #endif

            if (compact_paths) {
                cudaMemsetAsync(pt_state.dev_num_active_paths, 0, sizeof(int), 0);
            }

            shadePath<<<numblocksPathSegmentTracing, blockSize1d>>>(
                iter,
                num_paths,
                dev_paths,
                dev_path_indices,
                compact_paths ? dev_next_path_indices : nullptr,
                pt_state.dev_num_active_paths,
                pt_state.dev_materials,
                pt_state.dev_intersections,
                pt_state.dev_direct_light_intersections,
                pt_state.dev_environment_map_intersections,
                depth,
                !pt_state.hst_scene->exr_data.empty(),
                pt_state.exr_texture,
                TextureHandler::get().dev_textures,
                pt_state.hst_scene->emissive_geoms.size(),
                pt_state.dev_emissive_geoms,
                pt_state.dev_emissive_geom_area_prefix,
                pt_state.dev_geoms,
                pt_state.hst_scene->total_emissive_mesh_area,
                pt_state.dev_hdri_marginal_cdf,
                pt_state.dev_hdri_conditional_cdfs,
                pt_state.hst_scene->exr_width,
                pt_state.hst_scene->exr_height
            );

            CUDA_TIMER_RECORD(cudaTimer, "Shade + Compact, Iter {}", depth);

            if (compact_paths) {
                cudaMemcpy(&num_paths, pt_state.dev_num_active_paths, sizeof(int), cudaMemcpyDeviceToHost);
                std::swap(dev_path_indices, dev_next_path_indices);
                checkCUDAError("stream compaction");

                CUDA_TIMER_RECORD(cudaTimer, "Path Count Readback, Iter {}", depth);
            }
        }

        if (depth == traceDepth) {
            iterationComplete = true; // TODO: should be based off stream compaction results.
        }

        if (pt_state.guiData != NULL)
        {
            pt_state.guiData->TracedDepth = depth;
        }

        if (num_paths == 0) {
            iterationComplete = true;
        }

        CUDA_TIMER_RECORD(cudaTimer, "End, Iter {}", depth);

        if (PathTracerOptions::Get()->debug_bvh) {
            break;
        }
    }
    
    // cudaTimer.report();

    #if PROFILE
        if (pt_state.guiData != NULL)
        {
            pt_state.guiData->TimerBars = cudaTimer.stage_bars();
        }
    #endif

    // printf("Total Iteration Elapsed Time: %f\n\n", cudaTimer.get_elapsed("Start, Iter 1", "End, Iter 8"));

    cudaTimer.clean();

    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, pt_state.dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, float(iter), pt_state.dev_image);

    checkCUDAError("pathtrace");
}

