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
namespace rmhd1d { struct Conserved; }
namespace rmhd1d { struct Primitive; }
namespace rmhd2d { struct Conserved; }
namespace rmhd2d { struct Primitive; }
namespace rmhd3d { struct Conserved; }
namespace rmhd3d { struct Primitive; }

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
struct is_relativistic_mhd<rmhd1d::Conserved>
{
    static constexpr bool value = true;
};
template<>
struct is_relativistic_mhd<rmhd1d::Primitive>
{
    static constexpr bool value = true;
};
template<>
struct is_relativistic_mhd<rmhd2d::Conserved>
{
    static constexpr bool value = true;
};
template<>
struct is_relativistic_mhd<rmhd2d::Primitive>
{
    static constexpr bool value = true;
};
template<>
struct is_relativistic_mhd<rmhd3d::Conserved>
{
    static constexpr bool value = true;
};
template<>
struct is_relativistic_mhd<rmhd3d::Primitive>
{
    static constexpr bool value = true;
};


template <typename T>
struct is_1D_mhd_primitive {
  static const bool value = false;
};

template <>
struct is_1D_mhd_primitive<rmhd1d::Primitive> {
  static const bool value = true;
};

template <typename T>
struct is_2D_mhd_primitive {
  static const bool value = false;
};

template <>
struct is_2D_mhd_primitive<rmhd2d::Primitive> {
  static const bool value = true;
};

template <typename T>
struct is_3D_mhd_primitive {
  static const bool value = false;
};

template <>
struct is_3D_mhd_primitive<rmhd3d::Primitive> {
  static const bool value = true;
};

#endif