#include "app/render_imgui.h"

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

void InitImguiData(GuiDataContainer* guiData)
{
    AppState& app = AppState::Get();
    app.imguiData = guiData;
}

#if !UBER_SHADER
void imgui_material_label() {
    std::string s = "";
    AppState& app = AppState::Get();

    MaterialType mat_type = app.scene->materials[PathTracerOptions::Get()->selected_material].material_type;

    if (mat_type == MaterialType::Diffuse) {
        s = "Diffuse";
    } 
    else if (mat_type == MaterialType::Specular) {
        s = "Specular";
    } 
    else if (mat_type == MaterialType::Emissive) {
        s = "Emissive";
    } 
    else if (mat_type == MaterialType::Glass) {
        s = "Glass";
    } 
    else if (mat_type == MaterialType::Microfacet) {
        s = "Microfacet";
    }

    ImGui::Text("Material Type: %s", s);
}
#endif

// LOOK: Un-Comment to check ImGui Usage
void RenderImGui()
{
    AppState& app = AppState::Get();
    app.mouseOverImGuiWinow = app.io->WantCaptureMouse;

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("Path Tracer Analytics");

    ImGuiIO& io = ImGui::GetIO();
    io.FontGlobalScale = 1.2f;

    bool changed = false;

    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::Text("Traced Depth %d", app.imguiData->TracedDepth);
    
    #if !OPTIX
        changed |= ImGui::Checkbox("Debug BVH", &PathTracerOptions::Get()->debug_bvh);
    #endif

    const char* material_debug_modes[6] = {"Off", "Albedo", "World Normal", "Normal Map", "Roughness", "Metallic"};
    changed |= ImGui::Combo("Material Debug Mode", &PathTracerOptions::Get()->material_debug_mode, material_debug_modes, IM_ARRAYSIZE(material_debug_modes));

    const bool material_debug_active = PathTracerOptions::Get()->material_debug_mode != 0;

    const char* color_modes[3] = {"Reinhard", "AgX", "ACES"};
    ImGui::BeginDisabled(material_debug_active);
    changed |= ImGui::Combo("Tone Mapper", &PathTracerOptions::Get()->color_mode, color_modes, IM_ARRAYSIZE(color_modes));
    ImGui::EndDisabled();
    if (material_debug_active) {
        ImGui::TextDisabled("Tone mapping disabled while a material debug view is active");
    }

    changed |= ImGui::SliderFloat("Environment Map Intensity", &PathTracerOptions::Get()->envmap_intensity, 0.0f, 25.0f);

    if (!app.scene->emissive_geoms.empty()) {
        changed |= ImGui::Checkbox("Direct Light Sampling (MIS)", &PathTracerOptions::Get()->direct_light_sampling);
    }

    if (!app.scene->exr_data.empty()) {
        changed |= ImGui::Checkbox("Environment Map Importance Sampling (MIS)", &PathTracerOptions::Get()->environment_map_importance_sampling);
    }

    auto material_select_getter = [](void* data, int idx, const char** out_text) -> bool {
        auto& vec = *static_cast<std::vector<std::string>*>(data);
        if (idx < 0 || idx >= (int)vec.size()) {
            return false;
        }
        *out_text = vec[idx].c_str();
        return true;
    };

    changed |= ImGui::Combo("Material Select", &PathTracerOptions::Get()->selected_material, material_select_getter, &app.scene->material_names, (int)app.scene->material_names.size());

    #if !UBER_SHADER
        imgui_material_label();
    #endif

    changed |= ImGui::ColorEdit3("RGB", glm::value_ptr(app.scene->materials[PathTracerOptions::Get()->selected_material].color));
    changed |= ImGui::SliderFloat("Roughness", &app.scene->materials[PathTracerOptions::Get()->selected_material].roughness, 0.0f, 1.0f);
    changed |= ImGui::SliderFloat("Metallic", &app.scene->materials[PathTracerOptions::Get()->selected_material].metallic, 0.0f, 1.0f);

    changed |= ImGui::SliderFloat("IOR", &app.scene->materials[PathTracerOptions::Get()->selected_material].indexOfRefraction, 0.0f, 5.0f);
    
    changed |= ImGui::ColorEdit3("Emission Color", glm::value_ptr(app.scene->materials[PathTracerOptions::Get()->selected_material].emission.emission_color));
    changed |= ImGui::SliderFloat("Emission Strength", &app.scene->materials[PathTracerOptions::Get()->selected_material].emission.emission_strength, 0.0f, 10.0f);

    changed |= ImGui::SliderFloat("Transmission", &app.scene->materials[PathTracerOptions::Get()->selected_material].transmission, 0.0f, 1.0f);

    ImGui::End();

    if (changed) {
        app.scene->precompute_emissive_mesh_area();
        if (app.scene->total_emissive_mesh_area < EPSILON) {
            PathTracerOptions::Get()->direct_light_sampling = false;
        }

        app.camchanged = true;
    }

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

}

bool MouseOverImGuiWindow()
{
    AppState& app = AppState::Get();
    return app.mouseOverImGuiWinow;
}
