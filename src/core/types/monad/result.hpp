#ifndef RESULT_HPP
#define RESULT_HPP

#include <string>
#include <variant>

namespace simbi {
    struct Error {
        std::string message;
        explicit Error(std::string msg) : message(std::move(msg)) {}
    };
    template <typename T>
    class Result
    {
        bool success_;
        // succes value or error message
        std::variant<T, Error> data_;

      public:
        // ctor
        Result(bool success, std::variant<T, Error> data)
            : success_(success), data_(std::move(data))
        {
        }

        // move ctor
        Result(Result&& other) noexcept
            : success_(other.success_), data_(std::move(other.data_))
        {
            other.success_ = false;
        }

        // copy ctor
        Result(const Result& other)
            : success_(other.success_), data_(other.data_)
        {
        }

        // move assignment
        Result& operator=(Result&& other) noexcept
        {
            if (this != &other) {
                success_       = other.success_;
                data_          = std::move(other.data_);
                other.success_ = false;
            }
            return *this;
        }

        static Result<T> ok(T value)
        {
            return Result<T>(true, std::move(value));
        }

        static Result<T> error(std::string msg)
        {
            return Result<T>(false, Error(std::move(msg)));
        }

        // function application (map)
        template <typename Func>
        auto map(Func f) const -> Result<std::invoke_result_t<Func, T>>
        {
            if (success_) {
                return Result<std::invoke_result_t<Func, T>>::ok(
                    f(std::get<T>(data_))
                );
            }
            else {
                return Result<std::invoke_result_t<Func, T>>::error(
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
}   // namespace simbi

#endif
