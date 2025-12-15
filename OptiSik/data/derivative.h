#pragma once
#include "data/expression.h"

namespace OptiSik {

namespace {
template <typename T> void setGradient (Expression<T>& expr, const T& grad) {
    expr.setGradient(grad);
}
} // namespace

template <typename T, typename TFunctor, typename... TArgs>
T derivative (const TFunctor& functor, Expression<T>& wrt, TArgs&&... args) {
    setGradient(wrt, T (1));
    return functor(std::forward<TArgs>(args)...).evaluate().gradient ();
}
} // namespace OptiSik