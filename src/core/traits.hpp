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
#include <type_traits>

// template <int dim>
// struct SRHD;

// template <int dim>
// struct RMHD;

// template <int dim>
// struct NEWTONIAN;

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

template <typename F, typename T>
struct has_index_param {
  private:
    template <typename U>
    static auto test(int
    ) -> decltype(std::declval<U>()(std::declval<T>(), std::size_t{}), std::true_type{});

    template <typename>
    static auto test(...) -> std::false_type;

  public:
    static constexpr bool value = decltype(test<F>(0))::value;
};

template <typename T>
class Maybe;

template <typename T>
struct is_maybe {
    static const bool value = false;
};

template <typename T>
struct is_maybe<Maybe<T>> {
    static const bool value = true;
};

template <typename T>
inline constexpr bool is_maybe_v = is_maybe<T>::value;

// Check if type has value_type member
template <typename T, typename = void>
struct has_value_type : std::false_type {
};

template <typename T>
struct has_value_type<T, std::void_t<typename T::value_type>> : std::true_type {
};

// Get value_type safely
template <typename T, typename Enable = void>
struct get_value_type {
    using type = T;   // Fallback to T if no value_type
};

template <typename T>
struct get_value_type<T, std::enable_if_t<has_value_type<T>::value>> {
    using type = typename T::value_type;
};

// Helper alias
template <typename T>
using get_value_type_t = typename get_value_type<T>::type;

template <typename Array>
struct array_value_type {
    using type = typename std::decay_t<Array>::value_type;
};

template <typename Array>
struct array_raw_type {
    using type = typename std::decay_t<Array>::raw_type;
};

// Solver traits
// template <typename T>
// struct solver_traits;

// template <int dim>
// struct solver_traits<RMHD<dim>> {
//     static constexpr bool is_relativistic     = true;
//     static constexpr bool has_magnetic_fields = true;
// };

// template <int dim>
// struct solver_traits<SRHD<dim>> {
//     static constexpr bool is_relativistic     = true;
//     static constexpr bool has_magnetic_fields = false;
// };

#endif