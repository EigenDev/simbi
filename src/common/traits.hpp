/**
 * Where all of the type traits will live
 * 
*/
#ifndef TRAITS_HPP
#define TRAITS_HPP

namespace hydro1d { struct Primitive; }
namespace hydro2d { struct Primitive; }
namespace hydro3d { struct Primitive; }
namespace sr1d { struct Conserved; }
namespace sr1d { struct Primitive; }
namespace sr2d { struct Conserved; }
namespace sr2d { struct Primitive; }
namespace sr3d { struct Conserved; }
namespace sr3d { struct Primitive; }
namespace rmhd {
    template<int dim>
    struct AnyConserved; 
}

namespace rmhd { 
    template<int dim>
    struct AnyPrimitive; 
}

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

template <typename T>
struct is_relativistic_mhd {
  static const bool value = false;
};

template<>
struct is_relativistic_mhd<rmhd::AnyConserved<1>>
{
    static constexpr bool value = true;
};
template<>
struct is_relativistic_mhd<rmhd::AnyPrimitive<1>>
{
    static constexpr bool value = true;
};
template<>
struct is_relativistic_mhd<rmhd::AnyConserved<2>>
{
    static constexpr bool value = true;
};
template<>
struct is_relativistic_mhd<rmhd::AnyPrimitive<2>>
{
    static constexpr bool value = true;
};
template<>
struct is_relativistic_mhd<rmhd::AnyConserved<3>>
{
    static constexpr bool value = true;
};
template<>
struct is_relativistic_mhd<rmhd::AnyPrimitive<3>>
{
    static constexpr bool value = true;
};


template <typename T>
struct is_1D_mhd_primitive {
  static const bool value = false;
};

template <>
struct is_1D_mhd_primitive<rmhd::AnyPrimitive<1>> {
  static const bool value = true;
};

template <typename T>
struct is_2D_mhd_primitive {
  static const bool value = false;
};

template <>
struct is_2D_mhd_primitive<rmhd::AnyPrimitive<2>> {
  static const bool value = true;
};

template <typename T>
struct is_3D_mhd_primitive {
  static const bool value = false;
};

template <>
struct is_3D_mhd_primitive<rmhd::AnyPrimitive<3>> {
  static const bool value = true;
};

#endif