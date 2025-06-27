/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            traits.hpp
 *  * @brief           provides type traits for primitive and custom types
 *  * @details
 *  *
 *  * @version         0.8.0
 *  * @date            2025-02-26
 *  * @author          Marcus DuPont
 *  * @email           marcus.dupont@princeton.edu
 *  *
 *  *==============================================================================
 *  * @build           Requirements & Dependencies
 *  *==============================================================================
 *  * @requires        C++20
 *  * @depends         CUDA >= 11.0, HDF5 >= 1.12, OpenMP >= 4.5
 *  * @platform        Linux, MacOS
 *  * @parallel        GPU (CUDA, HIP), CPU (OpenMP)
 *  *
 *  *==============================================================================
 *  * @documentation   Reference & Notes
 *  *==============================================================================
 *  * @usage
 *  * @note
 *  * @warning
 *  * @todo
 *  * @bug
 *  * @performance
 *  *
 *  *==============================================================================
 *  * @testing        Quality Assurance
 *  *==============================================================================
 *  * @test
 *  * @benchmark
 *  * @validation
 *  *
 *  *==============================================================================
 *  * @history        Version History
 *  *==============================================================================
 *  * 2025-02-26      v0.8.0      Initial implementation
 *  *
 *  *==============================================================================
 *  * @copyright (C) 2025 Marcus DuPont. All rights reserved.
 *  *==============================================================================
 */
#ifndef TRAITS_HPP
#define TRAITS_HPP

#include <cstddef>
#include <functional>
#include <tuple>
#include <type_traits>

//==========================================================================
//                  PRIMITIVE TYPE TRAITS
//==========================================================================
template <typename T>
struct is_primitive {
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
struct is_mhd {
    static const bool value = false;
};

template <typename F, typename T>
struct has_index_param {
  private:
    template <typename U>
    static auto test(
        int
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

template <typename T>
inline constexpr bool is_primitive_v = is_primitive<T>::value;

template <typename T>
struct is_conserved {
    static const bool value = false;
};

template <typename T>
inline constexpr bool is_conserved_v = is_conserved<T>::value;

template <typename T>
inline constexpr bool is_mhd_v = is_mhd<T>::value;
template <typename T>
inline constexpr bool is_relativistic_mhd_v = is_relativistic_mhd<T>::value;

template <typename T>
struct function_traits;

// Specialization for regular function types
template <typename Ret, typename... Args>
struct function_traits<Ret(Args...)> {
    using return_type                  = Ret;
    using signature                    = Ret(Args...);
    static constexpr std::size_t arity = sizeof...(Args);

    // Get argument type by index
    template <std::size_t I>
    using arg_type = std::tuple_element_t<I, std::tuple<Args...>>;
};

// Specialization for function pointers
template <typename Ret, typename... Args>
struct function_traits<Ret (*)(Args...)> : function_traits<Ret(Args...)> {
};

// Specialization for member function pointers
template <typename Class, typename Ret, typename... Args>
struct function_traits<Ret (Class::*)(Args...)>
    : function_traits<Ret(Args...)> {
};

// Specialization for const member function pointers
template <typename Class, typename Ret, typename... Args>
struct function_traits<Ret (Class::*)(Args...) const>
    : function_traits<Ret(Args...)> {
};

// Specialization for std::function
template <typename Ret, typename... Args>
struct function_traits<std::function<Ret(Args...)>>
    : function_traits<Ret(Args...)> {
};
#endif
