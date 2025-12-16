#pragma once
#include "data/expression.h"

namespace OptiSik {

namespace {
template <typename T> void setGradient (Expression<T>& expr, const T& grad) {
    expr.setGradient (grad);
}
} // namespace

template <typename T, typename TFunctor, typename... TArgs>
T derivative (const TFunctor& functor, Expression<T>& wrt, TArgs&&... args) {
    setGradient (wrt, T (1));
    return functor (std::forward<TArgs> (args)...).gradient ();
}

template <size_t order = 1, typename T>
auto derivative (const Expression<T>& expression) {
    if constexpr (order == 0)
        return expressionValue (expression.value ());
    else if constexpr (order == 1)
        return expressionValue(expression.gradient ());
    else
        return derivative<order - 1> (expression.gradient ());
}

} // namespace OptiSik