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

template <size_t TOrder = 1, typename T>
auto derivative (const Expression<T>& expression) {
    static_assert(TOrder <=  ExpressionInfo<Expression<T>>::order, "Requested derivative order exceeds expression order");
    if constexpr (TOrder == 0)
        return expressionValue (expression.value ());
    else if constexpr (TOrder == 1)
        return expressionValue(expression.gradient ());
    else
        return derivative<TOrder - 1> (expression.gradient ());
}

} // namespace OptiSik