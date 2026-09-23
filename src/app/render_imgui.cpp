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
#include <cmath>
#include <utility>
#include <vector>
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

#if PROFILE
void RenderStageTimings()
{
    AppState& app = AppState::Get();
    const auto& latest = app.imguiData->TimerBars;

    static std::vector<std::vector<TimerStage>> smoothed;
    const float smoothing = 0.05f;

    smoothed.resize(latest.size());
    for (size_t b = 0; b < latest.size(); ++b) {
        bool same_stages = smoothed[b].size() == latest[b].size();
        for (size_t s = 0; same_stages && s < latest[b].size(); ++s) {
            same_stages = smoothed[b][s].name == latest[b][s].name;
        }

        if (!same_stages) {
            smoothed[b] = latest[b];
            continue;
        }

        for (size_t s = 0; s < latest[b].size(); ++s) {
            smoothed[b][s].ms += smoothing * (latest[b][s].ms - smoothed[b][s].ms);
        }
    }

    const auto& bars = smoothed;

    static std::vector<std::pair<std::string, ImU32>> stage_colors;
    auto color_for = [](const std::string& name) -> ImU32 {
        for (const auto& stage : stage_colors) {
            if (stage.first == name) {
                return stage.second;
            }
        }
        float hue = std::fmod(stage_colors.size() * 0.618034f, 1.0f);
        ImU32 color = ImColor::HSV(hue, 0.6f, 0.9f);
        stage_colors.push_back({ name, color });
        return color;
    };

    ImGui::Begin("Stage Timings");

    if (bars.empty()) {
        ImGui::TextDisabled("No timing data");
        ImGui::End();
        return;
    }

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    const float bar_height = ImGui::GetTextLineHeight() + 4.0f;

    for (size_t b = 0; b < bars.size(); ++b) {
        float total = 0.0f;
        for (const TimerStage& stage : bars[b]) {
            total += stage.ms;
        }

        ImGui::Text("Bounce %d  (%.3f ms)", (int)b + 1, total);

        ImVec2 origin = ImGui::GetCursorScreenPos();
        float width = ImGui::GetContentRegionAvail().x;
        ImGui::Dummy(ImVec2(width, bar_height));

        if (total <= 0.0f) {
            continue;
        }

        float x = origin.x;
        for (const TimerStage& stage : bars[b]) {
            float w = width * (stage.ms / total);
            ImVec2 seg_min(x, origin.y);
            ImVec2 seg_max(x + w, origin.y + bar_height);
            draw_list->AddRectFilled(seg_min, seg_max, color_for(stage.name));
            if (ImGui::IsMouseHoveringRect(seg_min, seg_max)) {
                ImGui::SetTooltip("%s\n%.3f ms (%.1f%%)", stage.name.c_str(), stage.ms, 100.0f * stage.ms / total);
            }
            x += w;
        }
    }

    ImGui::Separator();

    const float swatch = ImGui::GetTextLineHeight();
    for (const auto& stage : stage_colors) {
        ImVec2 p = ImGui::GetCursorScreenPos();
        draw_list->AddRectFilled(p, ImVec2(p.x + swatch, p.y + swatch), stage.second);
        ImGui::Dummy(ImVec2(swatch, swatch));
        ImGui::SameLine();
        ImGui::TextUnformatted(stage.first.c_str());
    }

    ImGui::End();
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

    ImGui::Checkbox("Lock Settings & Camera", &app.locked);
    if (app.locked) {
        ImGui::TextDisabled("Locked: settings and camera frozen, render will not restart");
    }

    ImGui::BeginDisabled(app.locked);

    #if !OPTIX
        changed |= ImGui::Checkbox("Debug BVH", &PathTracerOptions::Get()->debug_bvh);
    #endif

    const char* material_debug_modes[7] = {"Off", "Albedo", "World Normal", "Normal Map", "Roughness", "Metallic", "Alpha"};
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
    const char* alpha_modes[3] = {"Opaque", "Mask", "Blend"};
    int alpha_mode = app.scene->materials[PathTracerOptions::Get()->selected_material].alpha_mode;
    if (ImGui::Combo("Alpha Mode", &alpha_mode, alpha_modes, IM_ARRAYSIZE(alpha_modes))) {
        app.scene->materials[PathTracerOptions::Get()->selected_material].alpha_mode = static_cast<AlphaMode>(alpha_mode);
        changed = true;
    }
    changed |= ImGui::SliderFloat("Alpha", &app.scene->materials[PathTracerOptions::Get()->selected_material].alpha, 0.0f, 1.0f);
    ImGui::BeginDisabled(alpha_mode != ALPHA_MODE_MASK);
    changed |= ImGui::SliderFloat("Alpha Cutoff", &app.scene->materials[PathTracerOptions::Get()->selected_material].alpha_cutoff, 0.0f, 1.0f);
    ImGui::EndDisabled();
    changed |= ImGui::SliderFloat("Roughness", &app.scene->materials[PathTracerOptions::Get()->selected_material].roughness, 0.0f, 1.0f);
    changed |= ImGui::SliderFloat("Metallic", &app.scene->materials[PathTracerOptions::Get()->selected_material].metallic, 0.0f, 1.0f);

    changed |= ImGui::SliderFloat("IOR", &app.scene->materials[PathTracerOptions::Get()->selected_material].indexOfRefraction, 0.0f, 5.0f);
    
    changed |= ImGui::ColorEdit3("Emission Color", glm::value_ptr(app.scene->materials[PathTracerOptions::Get()->selected_material].emission.emission_color));
    changed |= ImGui::SliderFloat("Emission Strength", &app.scene->materials[PathTracerOptions::Get()->selected_material].emission.emission_strength, 0.0f, 10.0f);

    changed |= ImGui::SliderFloat("Transmission", &app.scene->materials[PathTracerOptions::Get()->selected_material].transmission, 0.0f, 1.0f);

    ImGui::EndDisabled();

    ImGui::End();

    #if PROFILE
        RenderStageTimings();
    #endif

    if (changed && !app.locked) {
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
