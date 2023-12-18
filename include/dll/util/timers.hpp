//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <chrono>

#ifndef DLL_NO_TIMERS

#include <iosfwd>
#include <iomanip>
#include <sstream>

#endif

namespace dll {

/*!
 * \brief Utility for a simple timer
 */
struct stop_timer {
    std::chrono::time_point<std::chrono::steady_clock> start_time; ///< The start time

    /*!
     * \brief Start the timer
     */
    void start() {
        start_time = std::chrono::steady_clock::now();
    }

    /*!
     * \brief Stop the timer and get the elapsed since start
     * \return elapsed time since start()
     */
    size_t stop() const {
        auto end = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(end - start_time).count();
    }
};

#ifdef DLL_NO_TIMERS

/*!
 * \brief Dump the values of the timer on the console.
 *
 * This has no effect if the timers were disabled.
 */
inline void dump_timers() {
    std::cout << "Timers have been disabled by defining DLL_NO_TIMERS" << std::endl;
}

/*!
 * \brief Dump all timers values to the console, with percentage of time from
 * the total.
 *
 * The total is the counter with the maximum total time
 */
inline void dump_timers_one() {
    std::cout << "Timers have been disabled by defining DLL_NO_TIMERS" << std::endl;
}

/*!
 * \brief Dump all timers values to the console in the form of a nice table.
 */
inline void dump_timers_pretty() {
    std::cout << "Timers have been disabled by defining DLL_NO_TIMERS" << std::endl;
}

struct auto_timer {
    auto_timer(const char* /*name*/) {}
};

struct unsafe_auto_timer {
    unsafe_auto_timer(const char* /*name*/) {}
};

#else

constexpr size_t max_timers = 128; ///< The maximum number of timers

/*!
 * \brief A timer
 */
struct timer_t {
    const char* name;             ///< The name of the timer
    std::atomic<size_t> count;    ///< The number of times it was incremented
    std::atomic<size_t> duration; ///< The total duration

    /*!
     * \brief Initialize an empty counter
     */
    timer_t()
            : name(nullptr), count(0), duration(0) {}

    /*!
     * \brief Copy a timer
     */
    timer_t(const timer_t& rhs)
            : name(rhs.name), count(rhs.count.load()), duration(rhs.duration.load()) {}

    /*!
     * \brief Copy assign a timer
     */
    timer_t& operator=(const timer_t& rhs) {
        if (&rhs != this) {
            name     = rhs.name;
            count    = rhs.count.load();
            duration = rhs.duration.load();
        }

        return *this;
    }

    /*!
     * \brief Move construct a timer
     */
    timer_t(timer_t&& rhs)
            : name(std::move(rhs.name)), count(rhs.count.load()), duration(rhs.duration.load()) {}

    /*!
     * \brief Move assign a timer
     */
    timer_t& operator=(timer_t&& rhs) {
        if (&rhs != this) {
            name     = std::move(rhs.name);
            count    = rhs.count.load();
            duration = rhs.duration.load();
        }

        return *this;
    }
};

/*!
 * \brief The structure holding all the timers
 */
struct timers_t {
    std::array<timer_t, max_timers> timers; ///< The timers
    std::mutex lock; ///< The lock to protect the timers

    /*!
     * \brief Reset the status of the timers
     */
    void reset(){
        std::lock_guard<std::mutex> l(lock);

        for(auto& timer : timers){
            timer.name = nullptr;
            timer.duration = 0;
            timer.count = 0;
        }
    }
};

/*!
 * \brief Get a reference to the timer structure
 */
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
 * \brief Dump the values of the timer on the console.
 *
 * This has no effect if the timers were disabled.
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

/*!
 * \brief Dump all timers values to the console in the form of a nice table.
 */
inline void dump_timers_pretty() {
    decltype(auto) timers = get_timers().timers;

    if(timers.empty()){
        std::cout << "No timers have been recorded!" << std::endl;
        return;
    }

    std::cout << std::endl;

    //Sort the timers by duration (DESC)
    std::sort(timers.begin(), timers.end(), [](auto& left, auto& right) {
        return left.duration > right.duration;
    });

    double total_duration = timers.front().duration.load();

    constexpr size_t columns = 5;

    std::string column_name[columns];
    column_name[0] = "%";
    column_name[1] = "Timer";
    column_name[2] = "Count";
    column_name[3] = "Total";
    column_name[4] = "Average";

    size_t column_length[columns];
    column_length[0] = 8;
    column_length[1] = column_name[1].size();
    column_length[2] = column_name[2].size();
    column_length[3] = column_name[3].size();
    column_length[4] = column_name[4].size();

    // Compute the width of each column
    for (decltype(auto) timer : timers) {
        if (timer.name) {
            size_t count = timer.count;
            size_t duration = timer.duration;

            column_length[1] = std::max(column_length[1], std::string(timer.name).size());
            column_length[2] = std::max(column_length[2], std::to_string(count).size());
            column_length[3] = std::max(column_length[3], duration_str(duration).size());
            column_length[4] = std::max(column_length[4], duration_str(duration / count).size());
        }
    }

    const size_t line_length = (columns + 1) * 1 + 2 + (columns - 1) * 2 + std::accumulate(column_length, column_length + columns, 0);

    std::cout << " " << std::string(line_length, '-') << '\n';

    printf(" | %-*s | %-*s | %-*s | %-*s | %-*s |\n",
        int(column_length[0]), column_name[0].c_str(),
        int(column_length[1]), column_name[1].c_str(),
        int(column_length[2]), column_name[2].c_str(),
        int(column_length[3]), column_name[3].c_str(),
        int(column_length[4]), column_name[4].c_str());

    std::cout << " " << std::string(line_length, '-') << '\n';

    // Print all the used timers
    for (decltype(auto) timer : timers) {
        if (timer.name) {
            size_t count = timer.count;
            size_t duration = timer.duration;

            printf(" | %*.3f%% | %-*s | %-*s | %-*s | %-*s |\n",
                int(column_length[0] - 1), 100.0 * (duration / double(total_duration)),
                int(column_length[1]), timer.name,
                int(column_length[2]), std::to_string(count).c_str(),
                int(column_length[3]), duration_str(duration).c_str(),
                int(column_length[4]), duration_str(duration / count).c_str());
        }
    }

    std::cout << " " << std::string(line_length, '-') << '\n';
}

/*!
 * \brief Automatic timer with RAII.
 */
struct auto_timer {
    const char* name;                               ///< The name of the timer
    std::chrono::time_point<std::chrono::steady_clock> start; ///< The start time
    std::chrono::time_point<std::chrono::steady_clock> end;   ///< The end time

    /*!
     * \brief Create an auto_timer witht the given name
     * \param name The name of the timer
     */
    auto_timer(const char* name) : name(name) {
        start = std::chrono::steady_clock::now();
    }

    /*!
     * \brief Destructs the timer, effectively incrementing the timer.
     */
    ~auto_timer() {
        end           = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        decltype(auto) timers = get_timers();

        // Try to increment without the lock

        for (decltype(auto) timer : timers.timers) {
            if (timer.name == name) {
                timer.duration += duration;
                ++timer.count;

                return;
            }
        }

        {
            std::lock_guard<std::mutex> lock(timers.lock);

            // Retry again to increment with lock

            for (decltype(auto) timer : timers.timers) {
                if (timer.name == name) {
                    timer.duration += duration;
                    ++timer.count;

                    return;
                }
            }

            // At this point the timer does not exist, create it

            for (decltype(auto) timer : timers.timers) {
                if (!timer.name) {
                    timer.name     = name;
                    timer.duration = duration;
                    timer.count    = 1;

                    return;
                }
            }
        }

        // If there are no more timers
        std::cerr << "Unable to register timer " << name << std::endl;
    }
};

/*!
 * \brief Automatic timer with RAII, without synchronization.
 */
struct unsafe_auto_timer {
    const char* name;                                         ///< The name of the timer
    std::chrono::time_point<std::chrono::steady_clock> start; ///< The start time
    std::chrono::time_point<std::chrono::steady_clock> end;   ///< The end time

    /*!
     * \brief Create an unsafe_auto_timer witht the given name
     * \param name The name of the timer
     */
    unsafe_auto_timer(const char* name) : name(name) {
        start = std::chrono::steady_clock::now();
    }

    /*!
     * \brief Destructs the timer, effectively incrementing the timer.
     */
    ~unsafe_auto_timer() {
        end           = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        decltype(auto) timers = get_timers();

        // Increment the timer if it exists

        for (decltype(auto) timer : timers.timers) {
            if (timer.name == name) {
                timer.duration += duration;
                ++timer.count;

                return;
            }
        }

        // At this point the timer does not exist, create it

        for (decltype(auto) timer : timers.timers) {
            if (!timer.name) {
                timer.name     = name;
                timer.duration = duration;
                timer.count    = 1;

                return;
            }
        }

        // If there are no more timers
        std::cerr << "Unable to register timer " << name << std::endl;
    }
};

#endif

} //end of namespace dll
