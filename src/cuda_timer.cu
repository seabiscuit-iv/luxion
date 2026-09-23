#include "cuda_timer.h"

void CudaTimer::clean() {
    for (auto& e : events) {
        cudaEventDestroy(e.event);
    }
    events.clear();
}

CudaTimer::~CudaTimer() {
    clean();
}

void CudaTimer::record(const std::string& name) {
    cudaEvent_t event;
    cudaEventCreate(&event);
    cudaEventRecord(event);
    events.push_back({
        name, event
    });
}

void CudaTimer::report() {
    if (events.size() < 2) {
        return;
    }

    cudaEventSynchronize(events.back().event);

    printf("\n===== CUDA Timer Report =====\n");

    for (size_t i = 1; i < events.size(); ++i) {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, events[i - 1].event, events[i].event);
        printf("[%s -> %s] : %f ms\n", events[i - 1].name.c_str(), events[i].name.c_str(), ms);
    }

    printf(  "=============================\n\n");
}

float CudaTimer::get_elapsed(const std::string& from, const std::string& to) {
    auto it1 = findEvent(from);
    auto it2 = findEvent(to);

    if (it1 == events.end() || it2 == events.end()) {
        return -1.0f;
    }

    cudaEventSynchronize(it2->event);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, it1->event, it2->event);
    return ms;
}

std::vector<std::vector<TimerStage>> CudaTimer::stage_bars() {
    std::vector<std::vector<TimerStage>> bars;
    if (events.size() < 2) {
        return bars;
    }

    cudaEventSynchronize(events.back().event);

    for (size_t i = 0; i < events.size(); ++i) {
        const std::string& name = events[i].name;

        if (name.rfind("Start", 0) == 0) {
            bars.emplace_back();
            continue;
        }
        if (i == 0 || bars.empty() || name.rfind("End", 0) == 0) {
            continue;
        }

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, events[i - 1].event, events[i].event);
        bars.back().push_back({ name.substr(0, name.find(", Iter")), ms });
    }

    return bars;
}
