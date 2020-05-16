//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#define UNIQUE_NAME_LINE2( name, line ) name##line
#define UNIQUE_NAME_LINE( name, line ) UNIQUE_NAME_LINE2( name, line )
#define UNIQUE_NAME( name ) UNIQUE_NAME_LINE( name, __LINE__ )

#define INTERNAL_DLL_TEMPLATE_TEST_CASE_DECL(name, description, T) \
    template <typename T>                                            \
    static void UNIQUE_NAME(____T_E_M_P_L_A_TE____T_E_S_T____)();    \
    TEST_CASE(name)

#define INTERNAL_DLL_TEMPLATE_TEST_CASE_SECTION(Tn)           \
    SUBCASE(#Tn) {                                            \
        UNIQUE_NAME(____T_E_M_P_L_A_TE____T_E_S_T____)<Tn>(); \
    }

#define INTERNAL_DLL_TEMPLATE_TEST_CASE_DEFN(T) \
    template <typename T>                         \
    static void UNIQUE_NAME(____T_E_M_P_L_A_TE____T_E_S_T____)()

#define TEMPLATE_TEST_CASE_2(name, description, T, T1, T2)         \
    INTERNAL_DLL_TEMPLATE_TEST_CASE_DECL(name, description, T) { \
        INTERNAL_DLL_TEMPLATE_TEST_CASE_SECTION(T1)              \
        INTERNAL_DLL_TEMPLATE_TEST_CASE_SECTION(T2)              \
    }                                                              \
    INTERNAL_DLL_TEMPLATE_TEST_CASE_DEFN(T)

#define TEMPLATE_TEST_CASE_4(name, description, T, T1, T2, T3, T4) \
    INTERNAL_DLL_TEMPLATE_TEST_CASE_DECL(name, description, T) { \
        INTERNAL_DLL_TEMPLATE_TEST_CASE_SECTION(T1)              \
        INTERNAL_DLL_TEMPLATE_TEST_CASE_SECTION(T2)              \
        INTERNAL_DLL_TEMPLATE_TEST_CASE_SECTION(T3)              \
        INTERNAL_DLL_TEMPLATE_TEST_CASE_SECTION(T4)              \
    }                                                              \
    INTERNAL_DLL_TEMPLATE_TEST_CASE_DEFN(T)
