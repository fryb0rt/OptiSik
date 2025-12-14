#pragma once

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

} // namespace OptiSik