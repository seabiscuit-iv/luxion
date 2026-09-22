#include "app/app_state.h"

#include "glslUtility.hpp"
#include "image.h"
#include "pathtrace.h"
#include "scene.h"
#include "sceneStructs.h"
#include "utilities.h"
#include "myoptix.h"
#include "config.h"

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "ImGui/imgui.h"
#include "ImGui/imgui_impl_glfw.h"
#include "ImGui/imgui_impl_opengl3.h"

#include "app/gl_resources.h"
#include "app/window.h"
#include "app/render_imgui.h"
#include "app/render_loop.h"
#include "app/save_image.h"
#include "app/input_callbacks.h"

#include <ctime>

void terminateHandler() {
    if (auto ex = std::current_exception()) {
        try {
            std::rethrow_exception(ex);
        } catch (const std::exception& e) {
            std::cerr << "Uncaught exception: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "Uncaught non-standard exception" << std::endl;
        }
    } else {
        std::cerr << "Terminate called without active exception" << std::endl;
    }
    std::abort();
}

std::string currentTimeString()
{
    time_t now;
    time(&now);
    char buf[sizeof "0000-00-00_00-00-00z"];
    strftime(buf, sizeof buf, "%Y-%m-%d_%H-%M-%Sz", gmtime(&now));
    return std::string(buf);
}


//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv)
{
    AppState& app = AppState::Get();
    std::set_terminate(terminateHandler);
    app.startTimeString = currentTimeString();

    const char* sceneFile = nullptr;
    const char* env_map_path = nullptr;
    unsigned int iterations = 5000;

    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], "-e") == 0 || strcmp(argv[i], "--envmap") == 0)
        {
            if (i + 1 >= argc)
            {
                printf("Error: %s requires a path\n", argv[i]);
                return 1;
            }
            env_map_path = argv[++i];
        }
        else if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--iterations") == 0)
        {
            if (i + 1 >= argc)
            {
                printf("Error: %s requires a number\n", argv[i]);
                return 1;
            }
            char* end = nullptr;
            long val = strtol(argv[++i], &end, 10);
            if (*end != '\0' || val <= 0)
            {
                printf("Error: invalid iteration count '%s'\n", argv[i]);
                return 1;
            }
            iterations = static_cast<unsigned int>(val);
        }
        else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0)
        {
            if (i + 1 >= argc)
            {
                printf("Error: %s requires a name\n", argv[i]);
                return 1;
            }
            app.outputName = argv[++i];
        }
        else
        {
            // Assume first non-flag argument is the scene file
            if (!sceneFile) {
                sceneFile = argv[i];
            }
            else
            {
                printf("Error: unexpected argument '%s'\n", argv[i]);
                return 1;
            }
        }
    }

    if (!sceneFile)
    {
        printf("Usage: %s SCENEFILE [-e|--envmap ENVMAP] [-i|--iterations N] [-o|--output NAME]\n", argv[0]);
        return 1;
    }

    //Create Instance for ImGUIData
    app.guiData = new GuiDataContainer();

    // Load scene file
    app.scene = new Scene(sceneFile, env_map_path);

    // Set up camera stuff from loaded path tracer settings
    app.iteration = 0;
    app.renderState = &app.scene->state;
    app.renderState->iterations = iterations;
    Camera& cam = app.renderState->camera;
    app.width = cam.resolution.x;
    app.height = cam.resolution.y;

    glm::vec3 view = cam.view;
    glm::vec3 up = cam.up;
    glm::vec3 right = glm::cross(view, up);
    up = glm::cross(right, view);

    // Ensure normalized view direction
    glm::vec3 v = glm::normalize(cam.view);
    // Horizontal angle (yaw) around Y axis
    // 0 when looking down -Z
    app.phi = atan2(v.x, -v.z);
    // Vertical angle (pitch) from +Y
    // 0 when looking straight up, π when looking straight down
    app.theta = acos(glm::clamp(v.y, -1.0f, 1.0f));
    app.zoom = glm::length(cam.position - cam.lookAt);
    app.refUp = glm::normalize(glm::vec3(0.0, 1.0, 0.0));

    // Initialize CUDA and GL components
    init();

    // Initialize ImGui Data
    InitImguiData(app.guiData);
    InitDataContainer(app.guiData);

    for(Geom &g : app.scene->geoms) {
        if (g.type == GeomType::MESH && g.mesh.h_valid) {
            bool copied = false;
            for(Geom &s : app.scene->geoms) {
                if (s.type == GeomType::MESH && s.mesh.h_valid && s.mesh.d_valid && s.mesh.label == g.mesh.label && s.materialid == g.materialid) {
                    g.mesh.make_mesh_device_copy(s.mesh);
                    copied = true;
                    break;
                }
            }
            
            if (copied) {
                continue;
            }

            g.mesh.make_mesh_device();
        }
    }

    TextureHandler::get().load_textures_on_device();

    OptixModule module = nullptr;
    OptixPipelineCompileOptions pipeline_compile_options = {};
    compile_pathtracing_optix_module(module, pipeline_compile_options); 

    OptixProgramGroup raygen_prog_group = nullptr;
    OptixProgramGroup miss_prog_group = nullptr;
    OptixProgramGroup directlight_miss_prog_group = nullptr;
    OptixProgramGroup hit_prog_group = nullptr;
    OptixProgramGroup directlight_hit_prog_group = nullptr;
    OptixProgramGroup envmap_hit_prog_group = nullptr;
    OptixProgramGroup envmap_miss_prog_group = nullptr;
    create_optix_program_groups(module, raygen_prog_group, miss_prog_group, directlight_miss_prog_group, hit_prog_group, directlight_hit_prog_group, envmap_miss_prog_group, envmap_hit_prog_group);

    OptixPipeline optix_pipeline;
    initialize_optix_pipeline(raygen_prog_group, miss_prog_group, directlight_miss_prog_group, hit_prog_group, directlight_hit_prog_group, envmap_miss_prog_group, envmap_hit_prog_group, pipeline_compile_options, optix_pipeline);

    OptixShaderBindingTable sbt = {};
    create_optix_sbt(sbt, raygen_prog_group, miss_prog_group, directlight_miss_prog_group, hit_prog_group, directlight_hit_prog_group, envmap_hit_prog_group, envmap_miss_prog_group);

    std::vector<OptixInstance> optix_instances;
    int id = 0;
    for(Geom &g : app.scene->geoms) {
        if (g.type == GeomType::MESH && g.mesh.d_valid) {
            // Create a single IAS from all GAS
            OptixInstance inst = {};

            float transform[12] = {
                g.transform[0][0], g.transform[1][0], g.transform[2][0], g.transform[3][0],
                g.transform[0][1], g.transform[1][1], g.transform[2][1], g.transform[3][1],
                g.transform[0][2], g.transform[1][2], g.transform[2][2], g.transform[3][2],
            };  

            memcpy(inst.transform, transform, sizeof(float) * 12);

            inst.instanceId = id;
            // inst.sbtOffset = id * RAY_TYPE_COUNT;
            inst.sbtOffset = 0;
            inst.visibilityMask = 255;
            inst.flags = OPTIX_INSTANCE_FLAG_NONE;
            inst.traversableHandle = g.mesh.as_handle;

            optix_instances.push_back(inst);

            id++;
        }
    }

    CUdeviceptr d_optix_instances;
    OptixTraversableHandle ias_handle;

    create_ias(optix_instances, d_optix_instances, ias_handle);

    app.scene->optix_pipeline = optix_pipeline;
    app.scene->ias_handle = ias_handle;
    app.scene->optix_sbt = sbt;

    // GLFW main loop
    mainLoop();

    for(Geom g : app.scene->geoms) {
        if (g.type == GeomType::MESH && g.mesh.d_valid) {
            g.mesh.delete_mesh_device();
        }
    }

    return 0;
}
