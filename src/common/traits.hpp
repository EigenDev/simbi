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

template <typename T>
struct is_relativistic {
  static const bool value = false;
};

template<>
struct is_relativistic<sr1d::Primitive>
{
    static constexpr bool value = true;
};
template<>
struct is_relativistic<sr1d::Conserved>
{
    static constexpr bool value = true;
};
template<>
struct is_relativistic<sr2d::Primitive>
{
    static constexpr bool value = true;
};
template<>
struct is_relativistic<sr2d::Conserved>
{
    static constexpr bool value = true;
};
template<>
struct is_relativistic<sr3d::Primitive>
{
    static constexpr bool value = true;
};
template<>
struct is_relativistic<sr3d::Conserved>
{
    static constexpr bool value = true;
};

// template<typename T>
// struct is_1D_class {
//     static const bool value = false;
// };
// template<>
// struct is_1D_class<simbi::Newtonian1D> {
//     static const bool value = true;
// };
// template<>
// struct is_1D_class<simbi::SRHD> {
//     static const bool value = true;
// };
// template<typename T>
// struct is_2D_class {
//     static const bool value = false;
// };
// template<>
// struct is_2D_class<simbi::SRHD2D> {
//     static const bool value = true;
// };
// template<>
// struct is_2D_class <simbi::Newtonian2D> {
//     static const bool value = true;
// };
// template<typename T>
// struct is_3D_class {
//     static const bool value = false;
// };
// template<>
// struct is_3D_class<simbi::SRHD3D> {
//     static const bool value = true;
// };
// template<>
// struct is_1D_class {
//     static const bool value = false;
// }
// template<>
// struct is_relativistic<simbi::SRHD>
// {
//     static constexpr bool value = true;
// };
// template<>
// struct is_relativistic<simbi::SRHD2D>
// {
//     static constexpr bool value = true;
// };
// template<>
// struct is_relativistic<simbi::SRHD3D>
// {
//     static constexpr bool value = true;
// }
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