#include "myoptix/optix_program_groups.h"
#include "myoptix/optix_check.h"
#include "myoptix/optix_context.h"

#include <optix_stubs.h>
#include <fmt/format.h>

void create_optix_program_groups(
    const OptixModule& module, 
    OptixProgramGroup& raygen_prog_group, 

    OptixProgramGroup& miss_prog_group, 
    OptixProgramGroup& directlight_miss_prog_group, 

    OptixProgramGroup& hitgroup_prog_group,
    OptixProgramGroup& out_directlight_prog_group,

    OptixProgramGroup& envmap_miss_prog_group,
    OptixProgramGroup& envmap_hit_prog_group
) {
    OptixDeviceContext optix = get_optix();

    OptixProgramGroupOptions program_group_options = {};

    OptixProgramGroupDesc raygen_prog_group_desc    = {}; //
    raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module            = module;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
    OPTIX_CHECK_LOG( optixProgramGroupCreate(
                optix,
                &raygen_prog_group_desc,
                1,   // num program groups
                &program_group_options,
                LOG, &LOG_SIZE,
                &raygen_prog_group
                ) );
    fmt::println("          Created Raygen Program");

    OptixProgramGroupDesc miss_prog_group_desc  = {};
    miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module            = module;
    miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
    OPTIX_CHECK_LOG( optixProgramGroupCreate(
                optix,
                &miss_prog_group_desc,
                1,   // num program groups
                &program_group_options,
                LOG, &LOG_SIZE,
                &miss_prog_group
                ) );
    fmt::println("          Created Miss Program");

    OptixProgramGroupDesc directlight_miss_prog_group_desc  = {};
    directlight_miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    directlight_miss_prog_group_desc.miss.module            = module;
    directlight_miss_prog_group_desc.miss.entryFunctionName = "__miss__ms_direct_light";
    OPTIX_CHECK_LOG( optixProgramGroupCreate(
                optix,
                &directlight_miss_prog_group_desc,
                1,   // num program groups
                &program_group_options,
                LOG, &LOG_SIZE,
                &directlight_miss_prog_group
                ) );
    fmt::println("          Created Direct Light Miss Program");

    OptixProgramGroupDesc hitgroup_prog_group_desc = {};
    hitgroup_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_prog_group_desc.hitgroup.moduleCH            = module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    hitgroup_prog_group_desc.hitgroup.moduleAH            = module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__ms_all";
    OPTIX_CHECK_LOG( optixProgramGroupCreate(
                optix,
                &hitgroup_prog_group_desc,
                1,   // num program groups
                &program_group_options,
                LOG, &LOG_SIZE,
                &hitgroup_prog_group
                ) );
    fmt::println("          Created Hit Program");

    OptixProgramGroupDesc directlight_hitgroup_prog_group_desc = {};
    directlight_hitgroup_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    directlight_hitgroup_prog_group_desc.hitgroup.moduleCH            = module;
    directlight_hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch_direct_light";
    directlight_hitgroup_prog_group_desc.hitgroup.moduleAH            = module;
    directlight_hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__ms_all";
    OPTIX_CHECK_LOG( optixProgramGroupCreate(
                optix,
                &directlight_hitgroup_prog_group_desc,
                1,   // num program groups
                &program_group_options,
                LOG, &LOG_SIZE,
                &out_directlight_prog_group
                ) );
    fmt::println("          Created Direct Light Hit Program");
    
    OptixProgramGroupDesc envmap_miss_prog_group_desc  = {};
    envmap_miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    envmap_miss_prog_group_desc.miss.module            = module;
    envmap_miss_prog_group_desc.miss.entryFunctionName = "__miss__ms_envmap";
    OPTIX_CHECK_LOG( optixProgramGroupCreate(
                optix,
                &envmap_miss_prog_group_desc,
                1,   // num program groups
                &program_group_options,
                LOG, &LOG_SIZE,
                &envmap_miss_prog_group
                ) );
    fmt::println("          Created Environment Map Miss Program");

    OptixProgramGroupDesc envmap_hit_prog_group_desc = {};
    envmap_hit_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    envmap_hit_prog_group_desc.hitgroup.moduleAH            = module;
    envmap_hit_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__ms_all";
    OPTIX_CHECK_LOG( optixProgramGroupCreate(
                optix,
                &envmap_hit_prog_group_desc,
                1,   // num program groups
                &program_group_options,
                LOG, &LOG_SIZE,
                &envmap_hit_prog_group
                ) );
    fmt::println("          Created Environment Map Hit Program");

    fmt::println("Optix Program Group Creation Complete");
}
