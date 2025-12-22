#pragma once

#include "data/vector.h"
#include <cmath>
#include <functional>

namespace OptiSik {

/// Simple gradient descent optimizer for unconstrained minimization
template <typename T, typename TVector = Vector<T>>
class GradientDescent {
public:
    struct Config {
        T learningRate       = T(0.01);
        size_t maxIterations = 1000;
        T tolerance          = T(1e-6);
    };

    struct Result {
        TVector x;
        T value;
        size_t iterations;
        bool converged;
        Result(TVector&& x, T value, size_t iterations, bool converged)
        : x(std::move(x)), value(value), iterations(iterations), converged(converged) {
        }
    };

    /// Minimize f(x) using gradient descent
    /// @param x0 Initial point
    /// @param objectiveFunc Function to minimize
    /// @param gradientFunc Gradient of the objective function
    /// @param config Configuration parameters
    template<typename TFunction, typename TGradient>
    static Result minimize(const TVector& x0,
                           const TFunction& objectiveFunc,
                           const TGradient& gradientFunc,
                           const Config& config = Config()) {
        TVector x = x0;
        for (size_t iter = 0; iter < config.maxIterations; ++iter) {
            const T fval       = objectiveFunc(x);
            const TVector grad = gradientFunc(x);

            // Compute gradient norm for convergence check
            T gradNorm = grad.magnitude();

            if (gradNorm < config.tolerance) {
                return Result(std::move(x), fval, iter + 1, true);
            }

            // Update step: x = x - learningRate * grad
            x -= config.learningRate * grad;
        }

        // Did not converge
        return Result(std::move(x), objectiveFunc(x), config.maxIterations, false);
    }
};

} // namespace OptiSik
