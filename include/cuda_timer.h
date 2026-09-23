#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>

#include <fmt/core.h>

#include "config.h"
#include "utilities.h"

#if PROFILE
    #define CUDA_TIMER_RECORD(timer, ...) (timer).record(fmt::format(__VA_ARGS__))
#else
    #define CUDA_TIMER_RECORD(timer, ...) ((void)0)
#endif

struct Event {
    std::string name;
    cudaEvent_t event;
};

struct CudaTimer {
    std::vector<Event> events;

    CudaTimer() = default;

    ~CudaTimer();
    void record(const std::string& name);
    void report();
    void clean();
    float get_elapsed(const std::string& from, const std::string& to);
    std::vector<std::vector<TimerStage>> stage_bars();


    auto findEvent(const std::string& name) {
        return std::find_if(events.begin(), events.end(),
            [&](const Event& e) { return e.name == name; });
    }

};