#ifndef RESULT_HPP
#define RESULT_HPP

#include <string>
#include <variant>

namespace simbi {
    template <typename T>
    class Result
    {
        bool success_;
        // succes value or error message
        std::variant<T, std::string> data_;

      public:
        static Result<T> ok(T value)
        {
            return Result<T>(true, std::move(value));
        }

        static Result<T> error(std::string msg)
        {
            return Result<T>(false, std::move(msg));
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
                    std::get<std::string>(data_)
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
                    std::get<std::string>(data_)
                );
            }
        }

        // accessors
        bool is_ok() const { return success_; }
        const T& value() const { return std::get<T>(data_); }
        const std::string& error() const
        {
            return std::get<std::string>(data_);
        }
    };
}   // namespace simbi

#endif
