#pragma once
#include <type_traits>

namespace OptiSik {

/// Helper struct for integer operations
template <typename T, typename TInteger> struct IntegerOperations {
    static T removeFraction (T value) {
        return T (TInteger (value));
    }
};

/// Helper struct for tolerance value
template <typename T> struct Tolerance {
    static constexpr T tolerance = T (1e-6f);
};

template<typename T> struct IsArithmeticType {
    static constexpr bool value = std::is_arithmetic_v<std::decay_t<T>>;
};

template<typename T>
constexpr bool IsArithmetic = IsArithmeticType<T>::value;

/// Integer exponent power function
template<size_t TExp, typename T, typename = std::enable_if_t<IsArithmetic<T>>>
T pow(T base) {
    if constexpr(TExp == 0) {
        return T(1);
    } else if constexpr(TExp == 1){
        return base;
    } else if constexpr(TExp % 2 == 0) {
        auto tmp = pow<TExp / 2>(base);
        return tmp * tmp;
    } else {
        return base * pow<TExp - 1>(base);
    }
}

} // namespace OptiSik