/**
 * ***********************(C) COPYRIGHT 2024 Marcus DuPont**********************
 * @file       traits.hpp
 * @brief      home of all type traits
 *
 * @note
 * @history:
 *   Version   Date            Author          Modification    Email
 *   V0.8.0    Dec-03-2023     Marcus DuPont marcus.dupont@princeton.edu
 *
 * @verbatim
 * ==============================================================================
 *
 * ==============================================================================
 * @endverbatim
 * ***********************(C) COPYRIGHT 2024 Marcus DuPont**********************
 */
#ifndef TRAITS_HPP
#define TRAITS_HPP

//==========================================================================
//                  PRIMITIVE TYPE TRAITS
//==========================================================================
template <typename T>
struct is_1D_primitive {
    static const bool value = false;
};

template <typename T>
struct is_2D_primitive {
    static const bool value = false;
};

template <typename T>
struct is_3D_primitive {
    static const bool value = false;
};

template <typename T>
struct is_relativistic {
    static const bool value = false;
};

template <typename T>
struct is_relativistic_mhd {
    static const bool value = false;
};

template <typename T>
struct is_1D_mhd_primitive {
    static const bool value = false;
};

template <typename T>
struct is_2D_mhd_primitive {
    static const bool value = false;
};

template <typename T>
struct is_3D_mhd_primitive {
    static const bool value = false;
};
#endif