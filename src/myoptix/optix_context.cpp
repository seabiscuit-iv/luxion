#include "myoptix/optix_context.h"
#include "myoptix/optix_check.h"

#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>

// only include this once
#include <optix_function_table_definition.h>

constexpr bool OPTIX_DEBUG_MODE = true;

bool optix_initialized = false;
OptixDeviceContext optix;

static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << level << "][" << tag << "]: "
              << message << "\n";
}

void init_optix() {
    cudaFree(0);
    try {
        OPTIX_CHECK(optixInit());
        OptixDeviceContextOptions optx_options;
        optx_options.logCallbackFunction = &context_log_cb;
        optx_options.logCallbackLevel = 4;
        optx_options.validationMode = OPTIX_DEBUG_MODE ? OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL : OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;
        CUcontext cu_ctx = 0;
        cuCtxGetCurrent(&cu_ctx);
        if (cu_ctx == nullptr) {
            printf("Error: No active CUDA context found!\n");
        }
        OPTIX_CHECK( optixDeviceContextCreate( cu_ctx, &optx_options, &optix ) );
    } 
    catch (const std::exception& e) {
        std::cerr << "OptiX initialization failed: " << e.what() << std::endl;
        exit(1);
    }
    std::cout << "OptiX initialized successfully!" << std::endl;
    optix_initialized = true;
}

OptixDeviceContext get_optix() {
    if (!optix_initialized) {
        std::cerr << "Attempted to call get_optix() before optix was initialized" << std::endl;
        exit(1);
    }
    return optix;
}
