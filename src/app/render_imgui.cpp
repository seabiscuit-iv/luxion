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
#include <cstdio>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <utility>
#include <vector>
#include "ImGui/imgui.h"

#if PROFILE
    #include <nvml.h>
#endif
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

    if (!app.showStageTimings) {
        return;
    }

    ImGui::Begin("Stage Timings", &app.showStageTimings);

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

#if PROFILE
void DrawUsageGraph(const char* label, const char* value_text, const float* history, int count, int offset, ImU32 color)
{
    ImGui::TextUnformatted(label);
    ImGui::SameLine(ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize(value_text).x);
    ImGui::TextUnformatted(value_text);

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const ImVec2 size(ImGui::GetContentRegionAvail().x, 90.0f);
    const ImVec2 corner(origin.x + size.x, origin.y + size.y);
    ImGui::Dummy(size);

    draw_list->AddRectFilled(origin, corner, ImGui::GetColorU32(ImGuiCol_FrameBg), 4.0f);

    const ImU32 grid_color = ImGui::GetColorU32(ImGuiCol_Border, 0.5f);
    for (int i = 1; i < 4; ++i) {
        float y = origin.y + size.y * (i / 4.0f);
        draw_list->AddLine(ImVec2(origin.x, y), ImVec2(corner.x, y), grid_color);
    }

    if (count < 2) {
        return;
    }

    ImVec4 base = ImGui::ColorConvertU32ToFloat4(color);
    const ImU32 fill_top = ImGui::ColorConvertFloat4ToU32(ImVec4(base.x, base.y, base.z, 0.45f));
    const ImU32 fill_bottom = ImGui::ColorConvertFloat4ToU32(ImVec4(base.x, base.y, base.z, 0.02f));
    const ImVec2 white_uv = ImGui::GetFontTexUvWhitePixel();

    std::vector<ImVec2> points(count);
    for (int i = 0; i < count; ++i) {
        float value = history[(offset + i) % count];
        value = value < 0.0f ? 0.0f : (value > 100.0f ? 100.0f : value);
        points[i] = ImVec2(
            origin.x + size.x * (float(i) / float(count - 1)),
            corner.y - size.y * (value / 100.0f)
        );
    }

    for (int i = 0; i + 1 < count; ++i) {
        const ImVec2& a = points[i];
        const ImVec2& b = points[i + 1];

        draw_list->PrimReserve(6, 4);
        ImDrawIdx idx = (ImDrawIdx)draw_list->_VtxCurrentIdx;
        draw_list->PrimWriteIdx(idx);
        draw_list->PrimWriteIdx(idx + 1);
        draw_list->PrimWriteIdx(idx + 2);
        draw_list->PrimWriteIdx(idx);
        draw_list->PrimWriteIdx(idx + 2);
        draw_list->PrimWriteIdx(idx + 3);
        draw_list->PrimWriteVtx(a, white_uv, fill_top);
        draw_list->PrimWriteVtx(b, white_uv, fill_top);
        draw_list->PrimWriteVtx(ImVec2(b.x, corner.y), white_uv, fill_bottom);
        draw_list->PrimWriteVtx(ImVec2(a.x, corner.y), white_uv, fill_bottom);
    }

    draw_list->AddPolyline(points.data(), count, color, ImDrawFlags_None, 2.0f);
    draw_list->AddCircleFilled(points.back(), 3.5f, color);
    draw_list->AddRect(origin, corner, ImGui::GetColorU32(ImGuiCol_Border), 4.0f);
}

void RenderGpuUtilization()
{
    AppState& app = AppState::Get();
    if (!app.showGpuUtilization) {
        return;
    }

    static bool nvml_tried = false;
    static bool nvml_ready = false;
    static nvmlDevice_t device = nullptr;
    if (!nvml_tried) {
        nvml_tried = true;
        int cuda_device = 0;
        char pci_bus_id[32] = {};
        nvml_ready = nvmlInit_v2() == NVML_SUCCESS
            && cudaGetDevice(&cuda_device) == cudaSuccess
            && cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), cuda_device) == cudaSuccess
            && nvmlDeviceGetHandleByPciBusId_v2(pci_bus_id, &device) == NVML_SUCCESS;
    }

    ImGui::Begin("GPU Utilization", &app.showGpuUtilization);

    if (!nvml_ready) {
        ImGui::TextDisabled("NVML unavailable");
        ImGui::End();
        return;
    }

    constexpr int history_size = 120;
    static float gpu_history[history_size] = {};
    static float vram_history[history_size] = {};
    static int history_offset = 0;
    static double last_sample_time = -1.0;
    static unsigned int gpu_percent = 0;
    static unsigned long long vram_used = 0;
    static unsigned long long vram_total = 0;

    double now = ImGui::GetTime();
    if (now - last_sample_time >= 0.1) {
        last_sample_time = now;

        nvmlUtilization_t utilization;
        if (nvmlDeviceGetUtilizationRates(device, &utilization) == NVML_SUCCESS) {
            gpu_percent = utilization.gpu;
        }

        nvmlMemory_t memory;
        if (nvmlDeviceGetMemoryInfo(device, &memory) == NVML_SUCCESS) {
            vram_used = memory.used;
            vram_total = memory.total;
        }

        gpu_history[history_offset] = float(gpu_percent);
        vram_history[history_offset] = vram_total > 0 ? 100.0f * float(vram_used) / float(vram_total) : 0.0f;
        history_offset = (history_offset + 1) % history_size;
    }

    const float gib = 1024.0f * 1024.0f * 1024.0f;
    char value_text[64];

    snprintf(value_text, sizeof(value_text), "%u%%", gpu_percent);
    DrawUsageGraph("GPU", value_text, gpu_history, history_size, history_offset, IM_COL32(64, 196, 160, 255));

    ImGui::Spacing();

    snprintf(value_text, sizeof(value_text), "%.2f / %.2f GiB", float(vram_used) / gib, float(vram_total) / gib);
    DrawUsageGraph("VRAM", value_text, vram_history, history_size, history_offset, IM_COL32(150, 120, 230, 255));

    ImGui::End();
}
#endif

bool RenderAnalytics()
{
    AppState& app = AppState::Get();
    if (!app.showAnalytics) {
        return false;
    }

    ImGui::Begin("Path Tracer Analytics", &app.showAnalytics);

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

    return changed;
}

// LOOK: Un-Comment to check ImGui Usage
void RenderImGui()
{
    AppState& app = AppState::Get();
    app.mouseOverImGuiWinow = app.io->WantCaptureMouse;

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    if (app.hideImGui) {
        app.mouseOverImGuiWinow = false;
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        return;
    }

    ImGuiIO& io = ImGui::GetIO();
    io.FontGlobalScale = 1.2f;

    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("General")) {
            ImGui::MenuItem("Analytics", nullptr, &app.showAnalytics);
            ImGui::EndMenu();
        }
        #if PROFILE
            if (ImGui::BeginMenu("Profiling")) {
                ImGui::MenuItem("Stage Timings", nullptr, &app.showStageTimings);
                ImGui::MenuItem("GPU Utilization", nullptr, &app.showGpuUtilization);
                ImGui::EndMenu();
            }
        #endif
        ImGui::EndMainMenuBar();
    }

    bool changed = RenderAnalytics();

    #if PROFILE
        RenderStageTimings();
        RenderGpuUtilization();
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
