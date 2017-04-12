//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#ifdef DLL_NO_TIMERS

namespace dll {

inline void dump_timers() {
    //No timers
}

struct auto_timer {
    auto_timer(const char* /*name*/) {}
};

} //end of namespace dll

#else

#include <chrono>
#include <iosfwd>
#include <iomanip>
#include <sstream>

namespace chrono = std::chrono;

namespace dll {

constexpr const std::size_t max_timers = 64;

struct timer_t {
    const char* name;
    std::atomic<std::size_t> count;
    std::atomic<std::size_t> duration;

    timer_t()
            : name(nullptr), count(0), duration(0) {}

    timer_t(const timer_t& rhs)
            : name(rhs.name), count(rhs.count.load()), duration(rhs.duration.load()) {}

    timer_t& operator=(const timer_t& rhs) {
        if (&rhs != this) {
            name     = rhs.name;
            count    = rhs.count.load();
            duration = rhs.duration.load();
        }

        return *this;
    }

    timer_t(timer_t&& rhs)
            : name(std::move(rhs.name)), count(rhs.count.load()), duration(rhs.duration.load()) {}

    timer_t& operator=(timer_t&& rhs) {
        if (&rhs != this) {
            name     = std::move(rhs.name);
            count    = rhs.count.load();
            duration = rhs.duration.load();
        }

        return *this;
    }
};

struct timers_t {
    std::array<timer_t, max_timers> timers;
    std::mutex lock;

    void reset(){
        std::lock_guard<std::mutex> l(lock);

        for(auto& timer : timers){
            timer.name = nullptr;
            timer.duration = 0;
            timer.count = 0;
        }

    }
};

inline timers_t& get_timers() {
    static timers_t timers;
    return timers;
}

inline std::string to_string_precision(double duration, int precision = 6) {
    std::ostringstream out;
    out << std::setprecision(precision) << duration;
    return out.str();
}

inline std::string duration_str(double duration, int precision = 6) {
    if (duration > 1000.0 * 1000.0 * 1000.0) {
        return to_string_precision(duration / (1000.0 * 1000.0 * 1000.0), precision) + "s";
    } else if (duration > 1000.0 * 1000.0) {
        return to_string_precision(duration / (1000.0 * 1000.0), precision) + "ms";
    } else if (duration > 1000.0) {
        return to_string_precision(duration / 1000.0, precision) + "us";
    } else {
        return to_string_precision(duration, precision) + "ns";
    }
}

/*!
 * \brief Reset all timers
 */
inline void reset_timers() {
    decltype(auto) timers = get_timers();
    timers.reset();
}

/*!
 * \brief Dump all timers values to the console.
 */
inline void dump_timers() {
    decltype(auto) timers = get_timers().timers;

    //Sort the timers by duration (DESC)
    std::sort(timers.begin(), timers.end(), [](auto& left, auto& right) {
        return left.duration > right.duration;
    });

    // Print all the used timers
    for (decltype(auto) timer : timers) {
        if (timer.name) {
            size_t count = timer.count;
            size_t duration = timer.duration;
            std::cout << timer.name << "(" << count << ") : "
                      << duration_str(duration)
                      << " (" << duration_str(duration / count) << ")" << std::endl;
        }
    }
}

/*!
 * \brief Dump all timers values to the console, with percentage of time from
 * the total.
 *
 * The total is the counter with the maximum total time
 */
inline void dump_timers_one() {
    decltype(auto) timers = get_timers().timers;

    if(timers.empty()){
        return;
    }

    //Sort the timers by duration (DESC)
    std::sort(timers.begin(), timers.end(), [](auto& left, auto& right) {
        return left.duration > right.duration;
    });

    double total_duration = timers.front().duration.load();

    // Print all the used timers
    for (decltype(auto) timer : timers) {
        if (timer.name) {
            size_t count = timer.count;
            size_t duration = timer.duration;
            std::cout << timer.name << "(" << count << ") : "
                      << duration_str(duration)
                      << " (" << 100.0 * (duration / total_duration) << "%, " << duration_str(duration / count) << ")" << std::endl;
        }
    }
}

struct stop_timer {
    chrono::time_point<chrono::steady_clock> start_time;

    stop_timer() = default;

    void start() {
        start_time = chrono::steady_clock::now();
    }

    std::size_t stop() const {
        auto end = chrono::steady_clock::now();
        return chrono::duration_cast<chrono::milliseconds>(end - start_time).count();
    }
};

struct auto_timer {
    const char* name;
    chrono::time_point<chrono::steady_clock> start;
    chrono::time_point<chrono::steady_clock> end;

    auto_timer(const char* name)
            : name(name) {
        start = chrono::steady_clock::now();
    }

    ~auto_timer() {
        end           = chrono::steady_clock::now();
        auto duration = chrono::duration_cast<chrono::nanoseconds>(end - start).count();

        decltype(auto) timers = get_timers();

        for (decltype(auto) timer : timers.timers) {
            if (timer.name == name) {
                timer.duration += duration;
                ++timer.count;

                return;
            }
        }

        std::lock_guard<std::mutex> lock(timers.lock);

        for (decltype(auto) timer : timers.timers) {
            if (timer.name == name) {
                timer.duration += duration;
                ++timer.count;

                return;
            }
        }

        for (decltype(auto) timer : timers.timers) {
            if (!timer.name) {
                timer.name     = name;
                timer.duration = duration;
                timer.count    = 1;

                return;
            }
        }

        std::cerr << "Unable to register timer " << name << std::endl;
    }
};

} //end of namespace dll

#endif
