#pragma once

#include "data/vector.h"

namespace OptiSik {

/// Common configuration for different optimizers
template <typename T>
struct OptimizationConfig {
    T learningRate       = T(0.01);
    size_t maxIterations = 1000;
    T tolerance          = T(1e-6);
};

/// Common result of different optimizers
template <typename T, typename TVector = Vector<T>>
struct OptimizationResult {
    TVector x;
    T value;
    size_t iterations;
    bool converged;
    Result(TVector&& x, T value, size_t iterations, bool converged)
    : x(std::move(x)), value(value), iterations(iterations), converged(converged) {
    }
};

} // namespace OptiSik
