//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <chrono>

namespace chrono = std::chrono;

namespace dll {

constexpr const std::size_t max_timers = 64;

struct timer_t {
    std::string name;
    std::size_t count = 0;
    std::size_t duration = 0;
    std::mutex lock;
};

inline std::array<timer_t, max_timers>& get_timers(){
    static std::array<timer_t, max_timers> timers;
    return timers;
}

inline std::string to_string_precision(double duration, int precision = 6){
    std::ostringstream out;
    out << std::setprecision(precision) << duration;
    return out.str();
}

inline std::string duration_str(double duration, int precision = 6){
    if(duration > 1000.0 * 1000.0 * 1000.0){
        return to_string_precision(duration / (1000.0 * 1000.0 * 1000.0), precision) + "s";
    } else if(duration > 1000.0 * 1000.0){
        return to_string_precision(duration / (1000.0 * 1000.0), precision) + "ms";
    } else if(duration > 1000.0){
        return to_string_precision(duration / 1000.0, precision) + "us";
    } else {
        return to_string_precision(duration, precision) + "ns";
    }
}

inline void dump_timers(){
    decltype(auto) timers = get_timers();
    for (std::size_t i = 0; i < max_timers; ++i) {
        decltype(auto) timer = timers[i];

        if (!timer.name.empty()) {
            std::cout << timer.name << "(" << timer.count << ") : " << duration_str(timer.duration) << std::endl;
        }
    }
}

struct auto_timer {
    std::string name;
    chrono::time_point<chrono::steady_clock> start;
    chrono::time_point<chrono::steady_clock> end;

    auto_timer(std::string name) : name(name) {
        start = chrono::steady_clock::now();
    }

    ~auto_timer(){
        end           = chrono::steady_clock::now();
        auto duration = chrono::duration_cast<chrono::nanoseconds>(end - start).count();

        decltype(auto) timers = get_timers();

        for (std::size_t tries = 0; tries < 3; ++tries) {
            for (std::size_t i = 0; i < max_timers; ++i) {
                decltype(auto) timer = timers[i];

                if (timer.name.empty()) {
                    std::lock_guard<std::mutex> lock(timer.lock);

                    //Make sure another thread did not modifiy it in the mean time
                    if (timer.name.empty()) {
                        timer.name     = name;
                        timer.duration = duration;
                        timer.count    = 1;

                        return;
                    }
                } else if (timer.name == name) {
                    std::lock_guard<std::mutex> lock(timer.lock);

                    timer.duration += duration;
                    ++timer.count;

                    return;
                }
            }
        }

        std::cerr << "Unable to register timer " << name << std::endl;
    }
};

} //end of namespace dll
