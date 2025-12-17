#pragma once
#include "data/expression.h"

namespace OptiSik {

namespace {

template <size_t TGradientIndex, typename TExpression>
void setGradient (TExpression& expr, typename ExpressionInfo<TExpression>::Type gradient) {
    if constexpr (TGradientIndex == 0) {
        expr.value() = typename ExpressionInfo<TExpression>::Type(gradient);
    } else if constexpr (TGradientIndex == 1) {
        expr.setGradient (typename ExpressionInfo<TExpression>::Type(gradient));
    } else {
        setGradient<TGradientIndex - 1> (expr.value (), gradient);
    }
}

template <size_t TOrder, typename... TArgs> class WithRespectToInternal {};

template <size_t TGradientIndex, typename TExpression, typename... TArgs>
class WithRespectToInternal<TGradientIndex, TExpression, TArgs...> {
    WithRespectToInternal<TGradientIndex + 1, TArgs...> mNext;

    TExpression& mExpr;
    public:
    WithRespectToInternal (TExpression& e, TArgs&... args)
    : mExpr (e), mNext (args...) {
        setGradient<TGradientIndex> (mExpr, typename ExpressionInfo<TExpression>::Type (1));
    }
    ~WithRespectToInternal () {
        setGradient<TGradientIndex> (mExpr, typename ExpressionInfo<TExpression>::Type (0));
    }

    private:
};

} // namespace

template <typename... TArgs> class WithRespectTo {
    private:
    WithRespectToInternal<1, TArgs...> mImpl;

    public:
    WithRespectTo (TArgs&... args) : mImpl (args...) {
    }
};

template <typename TFunctor, typename... TWithRespectToArgs, typename... TArgs>
auto computeDerivative (TFunctor&& functor, const WithRespectTo<TWithRespectToArgs...>& wrt, TArgs&&... args) {
    return functor (std::forward<TArgs> (args)...);
}

template <size_t TOrder = 1, typename TExpression>
auto getDerivative (TExpression&& expression) {
    static_assert (TOrder <= ExpressionInfo<std::decay_t<TExpression>>::order,
    "Requested derivative order exceeds expression order");
    if constexpr (TOrder == 0)  
        return expressionValue (expression.value ());
    else if constexpr (TOrder == 1)
        return expressionValue (expression.gradient ());
    else
        return getDerivative<TOrder - 1> (expression.gradient ());
}



} // namespace OptiSik