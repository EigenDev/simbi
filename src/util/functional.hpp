#ifndef FUNCTIONAL_HPP
#define FUNCTIONAL_HPP

namespace simbi {
    template <typename Signature>
    class Function;

    template <typename R, typename... Args>
    class Function<R(Args...)>
    {
      public:
        Function() : callable(nullptr) {}

        template <typename F>
        Function(F f) : callable(new CallableImpl<F>(f))
        {
        }

        Function(const Function& other)
            : callable(other.callable ? other.callable->clone() : nullptr)
        {
        }

        Function(Function&& other) noexcept : callable(other.callable)
        {
            other.callable = nullptr;
        }

        ~Function() { delete callable; }

        Function& operator=(const Function& other)
        {
            if (this != &other) {
                delete callable;
                callable = other.callable ? other.callable->clone() : nullptr;
            }
            return *this;
        }

        Function& operator=(Function&& other) noexcept
        {
            if (this != &other) {
                delete callable;
                callable       = other.callable;
                other.callable = nullptr;
            }
            return *this;
        }

        R operator()(Args... args) const
        {
            if (!callable) {
                throw std::bad_function_call();
            }
            return callable->invoke(std::forward<Args>(args)...);
        }

        explicit operator bool() const { return callable != nullptr; }

      private:
        struct CallableBase {
            virtual ~CallableBase()              = default;
            virtual R invoke(Args... args) const = 0;
            virtual CallableBase* clone() const  = 0;
        };

        template <typename F>
        struct CallableImpl : CallableBase {
            F f;

            CallableImpl(F f) : f(f) {}

            R invoke(Args... args) const override
            {
                return f(std::forward<Args>(args)...);
            }

            CallableBase* clone() const override { return new CallableImpl(f); }
        };

        CallableBase* callable;
    };
}   // namespace simbi

#endif