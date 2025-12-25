#pragma once

#include "data/derivative.h"
#include "data/lumatrix.h"
#include "data/vector.h"
#include <cmath>

namespace OptiSik {

/// Solver for finding roots of a function
template <typename T, typename TVector = Vector<T>>
class NewtonSolver {
public:
    using Config = OptimizationConfig<T>;
    struct Result = OptimizationResult<T, TVector>;

    template <typename TFunctions, typename TJacobianFunctor>
    static Result solve(const TVector& x0,
                        const TFunctions& functions,
                        const TJacobianFunctor& jacobianFunc,
                        const Config& config = Config()) {
        TVector x = x0;
        for (size_t iter = 0; iter < config.maxIterations; ++iter) {
            const auto functionsValues = functions(x);
            const auto mag             = functionsValues.magnitude();
            if (mag < config.tolerance) {
                return Result(std::move(x), iter, true);
            }
            const auto jacobian        = jacobianFunc(x);
            const auto jacobianInverse = LUMatrix(jacobian).invert();
            x                          = x - jacobianInverse * functionsValues;
        }

        // Did not converge
        return Result(std::move(x), config.maxIterations, false);
    }
};

} // namespace OptiSik
