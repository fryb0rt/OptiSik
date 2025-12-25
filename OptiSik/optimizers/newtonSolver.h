#pragma once

#include "data/derivative.h"
#include "data/vector.h"
#include <cmath>

namespace OptiSik {

/// Solver for finding roots of a function
template <typename T, typename TVector = Vector<T>>
class NewtonSolver {
public:
    using Config = OptimizationConfig<T>;
    using Result = OptimizationResult<T, TVector>;

    /// Minimize f(x) using gradient descent
    /// @param x0 Initial point
    /// @param objectiveFunc Function to minimize
    /// @param gradientFunc Gradient of the objective function
    /// @param config Configuration parameters
    template <typename TFunction, typename TGradient>
    static Result solve(const TVector& x0,
                        const TFunction& objectiveFunc,
                        const TGradient& gradientFunc,
                        const Config& config = Config()) {
        TVector x = x0;
        for (size_t iter = 0; iter < config.maxIterations; ++iter) {
            const T fval    = objectiveFunc(x);
            const auto grad = gradientFunc(x);

            // Compute gradient norm for convergence check
            T gradNorm = grad.magnitude();

            if (gradNorm < config.tolerance) {
                return Result(std::move(x), fval, iter + 1, true);
            }

            // Update step: x = x - learningRate * grad
            x -= TVector(config.learningRate * grad);
        }

        // Did not converge
        return Result(std::move(x), objectiveFunc(x), config.maxIterations, false);
    }

    /// Minimize f(x) using gradient descent with automatic differentiation
    /// TFunction should us Expression type of at least order = 1
    /// @param x0 Initial point
    /// @param objectiveFunc Function to minimize
    /// @param config Configuration parameters
    template <typename TFunction>
    static Result minimize(const TVector& x0,
                           const TFunction& objectiveFunc,
                           const Config& config = Config()) {
        static_assert(IsExpression<decltype(objectiveFunc(x0))>::value,
                      "objectiveFunc must accept and return Expression");
        return GradientDescent::minimize(
        x0,
        [&objectiveFunc](TVector& x) { return static_cast<T>(objectiveFunc(x)); },
        [&objectiveFunc](TVector& x) { return gradient(objectiveFunc, x); }, config);
    }
};

} // namespace OptiSik
