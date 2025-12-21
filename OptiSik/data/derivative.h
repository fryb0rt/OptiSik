#pragma once
#include "data/expression.h"
#include "data/matrix.h"
#include "data/vector.h"


namespace OptiSik {

namespace {

template <size_t TGradientIndex, typename TExpression>
void setGradient(TExpression& expr, typename ExpressionInfo<TExpression>::Type gradient) {
    if constexpr (TGradientIndex == 0) {
        expr.value() = typename ExpressionInfo<TExpression>::Type(gradient);
    } else if constexpr (TGradientIndex == 1) {
        expr.setGradient(typename ExpressionInfo<TExpression>::Type(gradient));
    } else {
        setGradient<TGradientIndex - 1>(expr.value(), gradient);
    }
}

template <typename TVector, typename TExpression>
void getGradient(TVector& out, const TExpression& expr, const size_t index = 0) {
    using EI = ExpressionInfo<TExpression>;
    if constexpr (EI::order > 0) {
        out[index] = static_cast<EI::Type>(expr.gradient());
        if (EI::order > index) {
            getGradient(out, expr.gradient(), index + 1);
        }
    }
}

template <size_t TGradientIndex, typename... TArgs>
class WithRespectToInternal {};

template <size_t TGradientIndex, typename TExpression, typename... TArgs>
class WithRespectToInternal<TGradientIndex, TExpression, TArgs...> {
    WithRespectToInternal<TGradientIndex + 1, TArgs...> mNext;

    TExpression&& mExpr;

public:
    WithRespectToInternal(TExpression&& e, TArgs&&... args)
    : mExpr(std::forward<TExpression>(e)), mNext(std::forward<TArgs>(args)...) {
        setGradient<TGradientIndex>(
        mExpr, typename ExpressionInfo<std::decay_t<TExpression>>::Type(1));
    }
    ~WithRespectToInternal() {
        setGradient<TGradientIndex>(
        mExpr, typename ExpressionInfo<std::decay_t<TExpression>>::Type(0));
    }

private:
};

} // namespace

template <typename... TArgs>
class WithRespectTo {
private:
    WithRespectToInternal<1, TArgs...> mImpl;

public:
    WithRespectTo(TArgs&&... args) : mImpl(std::forward<TArgs>(args)...) {
    }
};

template <typename... TArgs>
auto withRespectTo(TArgs&&... args) {
    return WithRespectTo<TArgs&&...>(std::forward<TArgs>(args)...);
}

template <typename TFunctor, typename... TWithRespectToArgs, typename... TArgs>
auto computeDerivative(TFunctor&& functor,
                       const WithRespectTo<TWithRespectToArgs...>& wrt,
                       TArgs&&... args) {
    return functor(std::forward<TArgs>(args)...);
}

template <size_t TOrder = 1, typename TExpression>
auto getDerivative(TExpression&& expression) {
    static_assert(TOrder <= ExpressionInfo<std::decay_t<TExpression>>::order,
                  "Requested derivative order exceeds expression order");
    if constexpr (TOrder == 0)
        return expressionValue(expression.value());
    else if constexpr (TOrder == 1)
        return expressionValue(expression.gradient());
    else
        return getDerivative<TOrder - 1>(expression.gradient());
}

template <typename TFunctor, typename... TWithRespectToArgs, typename... TArgs>
auto derivative(TFunctor&& functor,
                const WithRespectTo<TWithRespectToArgs...>& wrt,
                TArgs&&... args) {
    auto d   = functor(std::forward<TArgs>(args)...);
    using EI = ExpressionInfo<decltype(d)>;
    SVector<typename EI::Type, EI::order> result;
    getGradient(result, d);
    return result;
}

namespace {

template <size_t index, typename T, typename... TArgs>
auto&& get(T&& arg0, TArgs&&... args) {
    if constexpr (index == 0) {
        return std::forward<T>(arg0);
    } else {
        return get<index - 1>(std::forward<TArgs>(args)...);
    }
}

template <size_t TRowIndex, size_t TColumnIndex, typename THessian, typename TFunctor, typename... TArgs>
void hessianColumn(THessian& hessian, TFunctor&& functor, TArgs&&... args) {
    if constexpr (hessian.cols() > TColumnIndex) {
        {
            // We use separate block to make sure wrt is destroyed before computing next item in the matrix
            auto wrt = withRespectTo(get<TRowIndex>(std::forward<TArgs>(args)...),
                                     get<TColumnIndex>(std::forward<TArgs>(args)...));
            auto d = computeDerivative(functor, wrt, std::forward<TArgs>(args)...);
            hessian(TRowIndex, TColumnIndex) = getDerivative<2>(d);
        }
        hessianColumn<TRowIndex, TColumnIndex + 1>(hessian, functor,
                                                   std::forward<TArgs>(args)...);
    }
}

template <size_t TRowIndex, typename THessian, typename TFunctor, typename... TArgs>
void hessianRow(THessian& hessian, TFunctor&& functor, TArgs&&... args) {
    if constexpr (hessian.rows() > TRowIndex) {
        hessianColumn<TRowIndex, 0>(hessian, functor, std::forward<TArgs>(args)...);
        hessianRow<TRowIndex + 1>(hessian, functor, std::forward<TArgs>(args)...);
    }
}
} // namespace

template <typename TFunctor, typename... TArgs>
auto hessian(TFunctor&& functor, TArgs&&... args) {
    constexpr size_t N = sizeof...(TArgs);
    using ExprType = std::decay_t<decltype(functor(std::forward<TArgs>(args)...))>;
    using T = ExpressionInfo<ExprType>::Type;
    SMatrix<T, N, N> result;
    hessianRow<0>(result, functor, std::forward<TArgs>(args)...);
    return result;
}

template <typename TFunctor, typename TVector>
auto hessian(TFunctor&& functor, TVector&& vector) {
    constexpr size_t N = vectorSize<std::decay_t<TVector>>;
    using ExprType = std::decay_t<decltype(functor(std::forward<TVector>(vector)))>;
    using T = ExpressionInfo<ExprType>::Type;
    SMatrix<T, N, N> result;
    for (size_t r = 0; r < N; ++r) {
        for (size_t c = 0; c < N; ++c) {
            using VectorType = typename std::decay_t<TVector>::Type;
            auto wrt = withRespectTo(std::forward<VectorType>(vector[r]), std::forward<VectorType>(vector[c]));
            auto d = computeDerivative(functor, wrt, std::forward<TVector>(vector));
            result(r, c) = getDerivative<2>(d);
        }
    }
    return result;
}

namespace {
template <size_t TFunctionIndex, typename... TFunctors>
class FunctionsInternal {};

template <size_t TFunctionIndex, typename TFunctor, typename... TFunctors>
class FunctionsInternal<TFunctionIndex, TFunctor, TFunctors...> {
    FunctionsInternal<TFunctionIndex + 1, TFunctors...> mNext;

    const TFunctor& mFunctor;

public:
    FunctionsInternal(const TFunctor& functor, const TFunctors&... functors)
    : mFunctor(functor), mNext(functors...) {
    }

    template <size_t TIndex, typename... TArgs>
    auto compute(TArgs&&... args) const {
        if constexpr (TIndex == TFunctionIndex) {
            return mFunctor(std::forward<TArgs>(args)...);
        } else {
            return mNext.template compute<TIndex>(std::forward<TArgs>(args)...);
        }
    }
};

} // namespace

template <typename... TFunctors>
class Functions {
private:
    FunctionsInternal<0, TFunctors...> mImpl;

public:
    Functions(const TFunctors&... functors) : mImpl(functors...) {
    }

    template <size_t TIndex, typename... TArgs>
    auto compute(TArgs&&... args) const {
        return mImpl.template compute<TIndex>(std::forward<TArgs>(args)...);
    }
};

namespace {
template <size_t TRowIndex, size_t TColumnIndex, typename TJacobian, typename... TFunctors, typename... TArgs>
void jacobianColumn(TJacobian& jacobian, const Functions<TFunctors...>& functions, TArgs&&... args) {
    if constexpr (jacobian.cols() > TColumnIndex) {
        {
            // We use separate block to make sure wrt is destroyed before computing next item in the matrix
            auto wrt = withRespectTo(get<TColumnIndex>(std::forward<TArgs>(args)...));
            auto d = functions.template compute<TRowIndex>(std::forward<TArgs>(args)...);
            jacobian(TRowIndex, TColumnIndex) = getDerivative<1>(d);
        }
        jacobianColumn<TRowIndex, TColumnIndex + 1>(jacobian, functions,
                                                    std::forward<TArgs>(args)...);
    }
}

template <size_t TRowIndex, typename TJacobian, typename... TFunctors, typename... TArgs>
void jacobianRow(TJacobian& jacobian, const Functions<TFunctors...>& functions, TArgs&&... args) {
    if constexpr (jacobian.rows() > TRowIndex) {
        jacobianColumn<TRowIndex, 0>(jacobian, functions, std::forward<TArgs>(args)...);
        jacobianRow<TRowIndex + 1>(jacobian, functions, std::forward<TArgs>(args)...);
    }
}

template <size_t TRowIndex, typename TJacobian, typename... TFunctors, typename TVector>
void jacobianRowVector(TJacobian& jacobian,
                       const Functions<TFunctors...>& functions,
                       TVector&& vector) {
    if constexpr (jacobian.rows() > TRowIndex) {
        for (size_t c = 0; c < jacobian.cols(); ++c) {
            auto wrt = withRespectTo(
            std::forward<typename std::decay_t<TVector>::Type>(vector[c]));
            auto d = functions.template compute<TRowIndex>(std::forward<TVector>(vector));
            jacobian(TRowIndex, c) = getDerivative<1>(d);
        }
        jacobianRowVector<TRowIndex + 1>(jacobian, functions, std::forward<TVector>(vector));
    }
}
} // namespace

template <typename... TFunctors, typename... TArgs>
auto jacobian(const Functions<TFunctors...>& functions, TArgs&&... args) {
    constexpr size_t cols = sizeof...(TArgs);
    constexpr size_t rows = sizeof...(TFunctors);
    using ExprType =
    std::decay_t<decltype(functions.template compute<0>(std::forward<TArgs>(args)...))>;
    using T = ExpressionInfo<ExprType>::Type;
    SMatrix<T, rows, cols> result;
    jacobianRow<0>(result, functions, std::forward<TArgs>(args)...);
    return result;
}

template <typename... TFunctors, typename TVector>
auto jacobian(const Functions<TFunctors...>& functions, TVector&& vector) {
    constexpr size_t cols = vectorSize<std::decay_t<TVector>>;
    constexpr size_t rows = sizeof...(TFunctors);
    using ExprType =
    std::decay_t<decltype(functions.template compute<0>(std::forward<TVector>(vector)))>;
    using T = ExpressionInfo<ExprType>::Type;
    SMatrix<T, rows, cols> result;
    jacobianRowVector<0>(result, functions, std::forward<TVector>(vector));
    return result;
}

template <typename TFunctor, typename... TArgs>
auto gradient(TFunctor&& functor, TArgs&&... args) {
    return jacobian(Functions(functor), std::forward<TArgs>(args)...)[0];
}

// template <typename TFunctor, typename TVector>
// auto gradient(TFunctor&& functor, TVector&& vector) {
//     struct NewFunctor
//     return apply(
//     [&functor](auto&&... args) {
//         return jacobian(Functions(functor), std::forward<TArgs>(args)...)[0];
//     },
//     std::forward<TFunctor>(vector))
// }

} // namespace OptiSik