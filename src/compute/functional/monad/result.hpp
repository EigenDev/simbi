#ifndef RESULT_HPP
#define RESULT_HPP

#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

namespace simbi {
    struct Error {
        std::string message;
        explicit Error(std::string msg) : message(std::move(msg)) {}
    };
    template <typename T>
    class result_t
    {
        bool success_;
        // succes value or error message
        std::variant<T, Error> data_;

      public:
        // ctor
        result_t(bool success, std::variant<T, Error> data)
            : success_(success), data_(std::move(data))
        {
        }

        // move ctor
        result_t(result_t&& other) noexcept
            : success_(other.success_), data_(std::move(other.data_))
        {
            other.success_ = false;
        }

        // copy ctor
        result_t(const result_t& other)
            : success_(other.success_), data_(other.data_)
        {
        }

        // move assignment
        result_t& operator=(result_t&& other) noexcept
        {
            if (this != &other) {
                success_       = other.success_;
                data_          = std::move(other.data_);
                other.success_ = false;
            }
            return *this;
        }

        static result_t<T> ok(T value)
        {
            return result_t<T>(true, std::move(value));
        }

        static result_t<T> error(std::string msg)
        {
            return result_t<T>(false, Error(std::move(msg)));
        }

        // function application (map)
        template <typename Func>
        auto map(Func f) const -> result_t<std::invoke_result_t<Func, T>>
        {
            if (success_) {
                return result_t<std::invoke_result_t<Func, T>>::ok(
                    f(std::get<T>(data_))
                );
            }
            else {
                return result_t<std::invoke_result_t<Func, T>>::error(
                    std::get<Error>(data_).message
                );
            }
        }

        // monadic bind (flat_map)
        template <typename Func>
        auto and_then(Func f) const -> std::invoke_result_t<Func, T>
        {
            if (success_) {
                return f(std::get<T>(data_));
            }
            else {
                return std::invoke_result_t<Func, T>::error(
                    std::get<Error>(data_).message
                );
            }
        }

        // accessors
        bool is_ok() const { return success_; }
        const T& value() const { return std::get<T>(data_); }
        const std::string& error() const
        {
            return std::get<Error>(data_).message;
        }
    };

    // add this specialization to your result.hpp
    template <>
    class result_t<void>
    {
        bool success_;
        std::optional<Error> error_;

      public:
        result_t(bool success, std::optional<Error> error = std::nullopt)
            : success_(success), error_(std::move(error))
        {
        }

        static result_t<void> ok()
        {
            return result_t<void>(true, std::nullopt);
        }

        static result_t<void> error(std::string msg)
        {
            return result_t<void>(false, Error(std::move(msg)));
        }

        bool is_ok() const { return success_; }
        const std::string& error() const { return error_->message; }

        // and_then for void results
        template <typename Func>
        auto and_then(Func f) const
        {
            if (success_) {
                return f();
            }
            else {
                using result_type = std::invoke_result_t<Func>;
                return result_type::error(error_->message);
            }
        }
    };
}   // namespace simbi

#endif
