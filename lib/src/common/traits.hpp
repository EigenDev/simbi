/**
 * Where all of the type traits will live
 * 
*/
#ifndef TRAITS_HPP
#define TRAITS_HPP

#include "hydro_structs.hpp"


//==========================================================================
//                  PRIMTIIVE TYPE TRAITS
//==========================================================================
template <typename T>
struct is_1D_primitive {
  static const bool value = false;
};

template<>
struct is_1D_primitive<hydro1d::Primitive>
{
    static constexpr bool value = true;
};

template<>
struct is_1D_primitive<sr1d::Primitive>
{
    static constexpr bool value = true;
};

template<typename T>
struct is_2D_primitive {
    static const bool value = false;
};

template<>
struct is_2D_primitive<hydro2d::Primitive>
{
    static constexpr bool value = true;
};

template<>
struct is_2D_primitive<sr2d::Primitive>
{
    static constexpr bool value = true;
};

template<typename T>
struct is_3D_primitive {
    static const bool value = false;
};

template<>
struct is_3D_primitive<hydro3d::Primitive>
{
    static constexpr bool value = true;
};

template<>
struct is_3D_primitive<sr3d::Primitive>
{
    static constexpr bool value = true;
};

// template<typename T>
// struct is_simstate {
//     static const bool value = false;
// };

// template<>
// struct is_simstate<simbi::SRHD> {
//     static const bool value = true;
// };

// template<>
// struct is_simstate<simbi::SRHD2D> {
//     static const bool value = true;
// };

// template<>
// struct is_simstate<simbi::SRHD3D> {
//     static const bool value = true;
// };


#endif