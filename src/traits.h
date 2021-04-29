/**
 * Where all of the type traits will live
 * 
*/
#include "hydro_structs.h"


//==========================================================================
//                  PRIMTIIVE TYPE TRAITS
//==========================================================================
template <typename T>
struct is_primitive {
  static const bool value = false;
};

template<>
struct is_primitive<hydro1d::Primitive>
{
    static constexpr bool value = true;
};

template<>
struct is_primitive<hydro2d::Primitive>
{
    static constexpr bool value = true;
};

template<>
struct is_primitive<sr1d::Primitive>
{
    static constexpr bool value = true;
};

template<>
struct is_primitive<sr2d::Primitive>
{
    static constexpr bool value = true;
};
