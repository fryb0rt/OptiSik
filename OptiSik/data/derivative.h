#pragma once
#include "data/expression.h"
#include "data/vector.h"

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

template<typename TVector, typename TExpression>
void getGradient(TVector& out, const TExpression& expr, const size_t index = 0) {
    using EI = ExpressionInfo<TExpression>;
    if constexpr(EI::order > 0) {
        out[index] = static_cast<EI::Type>(expr.gradient());
        if (EI::order > index) {
            getGradient(out, expr.gradient(), index + 1);
        }
    }
}

template <size_t TGradientIndex, typename... TArgs> class WithRespectToInternal {
};

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

template <typename TFunctor, typename... TWithRespectToArgs, typename... TArgs>
auto derivative(TFunctor&& functor, const WithRespectTo<TWithRespectToArgs...>& wrt, TArgs&&... args) {
    auto d = functor (std::forward<TArgs> (args)...);
    using EI = ExpressionInfo<decltype(d)>;
    SVector<typename EI::Type,EI::order> result;
    getGradient(result, d);
    return result;
}

template <typename TFunctor, typename... TArgs>
auto gradient(TFunctor&& functor, TArgs&... args) {
    WithRespectTo<TArgs...> wrt(args...);
    return derivative(functor, wrt, args...);
}

} // namespace OptiSik