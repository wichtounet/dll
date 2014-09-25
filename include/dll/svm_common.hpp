//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_SVM_COMMON
#define DLL_SVM_COMMON

//SVM Support is optional cause it requires libsvm

#ifdef DLL_SVM_SUPPORT

#include "nice_svm.hpp"

namespace dll {

inline svm_parameter default_svm_parameters(){
    auto parameters = svm::default_parameters();

    parameters.svm_type = C_SVC;
    parameters.kernel_type = RBF;
    parameters.probability = 1;
    parameters.C = 2.8;
    parameters.gamma = 0.0073;

    return parameters;
}

template<typename DBN>
void svm_store(const DBN& dbn, std::ostream& os){
    if(dbn.svm_loaded){
        binary_write(os, true);

        svm::save(dbn.svm_model, "..tmp.svm");

        std::ifstream svm_is("..tmp.svm", std::ios::binary);

        char buffer[1024];

        while(true){
            svm_is.read(buffer, 1024);

            if(svm_is.gcount() == 0){
                break;
            }

            os.write(buffer, svm_is.gcount());
        }
    } else {
        binary_write(os, false);
    }
}

template<typename DBN>
void svm_load(DBN& dbn, std::istream& is){
    dbn.svm_loaded = false;

    if(is.good()){
        bool svm;
        binary_load(is, svm);

        if(svm){
            std::ofstream svm_os("..tmp.svm", std::ios::binary);

            char buffer[1024];

            while(true){
                is.read(buffer, 1024);

                if(is.gcount() ==0){
                    break;
                }

                svm_os.write(buffer, is.gcount());
            }

            svm_os.close();

            dbn.svm_model = svm::load("..tmp.svm");

            dbn.svm_loaded = true;
        }
    }
}

} // end of namespace dll

#endif //DLL_SVM_SUPPORT

#endif //DLL_SVM_COMMON
