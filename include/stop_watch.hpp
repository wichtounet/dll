//=======================================================================
// Copyright Baptiste Wicht 2014.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//=======================================================================

#ifndef STOP_WATCH_HPP
#define STOP_WATCH_HPP

#include <chrono>

typedef std::chrono::high_resolution_clock clock_type;

template<typename precision = std::chrono::milliseconds>
class stop_watch {
public:
    stop_watch(){
        start_point = clock_type::now();
    }

    double elapsed(){
        auto end_point = clock_type::now();
        auto time = std::chrono::duration_cast<precision>(end_point - start_point);
        return time.count();
    }

private:
    clock_type::time_point start_point;
};

#endif
