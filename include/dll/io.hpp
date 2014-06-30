//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DBN_IO_HPP
#define DBN_IO_HPP

namespace dll {

//Binary I/O utility functions

template<typename T>
void binary_write(std::ostream& os, const T& v){
    os.write(reinterpret_cast<const char*>(&v), sizeof(v));
}

template<typename Container>
void binary_write_all(std::ostream& os, const Container& c){
    for(auto& v : c){
        binary_write(os, v);
    }
}

template<typename T>
void binary_load(std::istream& is, T& v){
    is.read(reinterpret_cast<char*>(&v), sizeof(v));
}

template<typename Container>
void binary_load_all(std::istream& is, Container& c){
    for(auto& v : c){
        binary_load(is, v);
    }
}

} //end of dbn namespace

#endif