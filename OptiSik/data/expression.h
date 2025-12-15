#pragma once

namespace OptiSik {

template <typename T> class Expression {
    T mValue;
    T mGrad;

    public:
    using TValueType = T;
    Expression() : mValue (T(0)), mGrad (T(0)) {}

    template<typename U>
    Expression (U&& value) {
        *this = std::forward<U>(value);
    }
    
    template<typename U>
    Expression (U&& value, U&& grad) : mValue (std::forward<U>(value)), mGrad (std::forward<U>(grad)) {
    }

    template<typename U>
    Expression& operator=(U&& value) {
        if constexpr (std::is_same_v<std::decay_t<U>, Expression<T>>) {
            mValue = value.value ();
            mGrad = value.gradient ();
        } else {
            mValue = std::forward<U>(value);
            mGrad  = T(0);
        }
        return *this;
    }

    operator T () const {
        return mValue;
    }
    template<typename U>
    operator U () const {
        return static_cast<U> (mValue);
    }
    Expression<T> evaluate () const {
        return *this;
    }
    T gradient () const {
        return mGrad;
    }
    T value () const {
        return mValue;
    }
    void setGradient (const T& grad) {
        mGrad = grad;
    }
};

template <typename T, typename TInput>
Expression<T> evaluate (const TInput& input) {
    if constexpr (std::is_arithmetic_v<TInput>)
        return Expression<T> (input);
    else
        return input.evaluate ();
}

template <typename TLeft, typename TRight>
class GetType{
public:
    using TValueType = typename TRight::TValueType;
};

template<template<typename> typename Expression, typename TRight, typename T>
class GetType<Expression<T>, TRight> {
public:
    using TValueType = T;
};

template <typename TLeft, typename TRight, typename T = GetType<TLeft, TRight>::TValueType>
Expression<T> operator+ (const TLeft& left, const TRight& right) {
    Expression<T> leftExpr  = OptiSik::evaluate<T, TLeft> (left);
    Expression<T> rightExpr = OptiSik::evaluate<T, TRight> (right);
    return Expression<T> (leftExpr.value () + rightExpr.value (),
    leftExpr.gradient () + rightExpr.gradient ());
}

template <typename TLeft, typename TRight, typename T = GetType<TLeft, TRight>::TValueType>
Expression<T> operator- (const TLeft& left, const TRight& right) {
    Expression<T> leftExpr  = OptiSik::evaluate<T, TLeft> (left);
    Expression<T> rightExpr = OptiSik::evaluate<T, TRight> (right);
    return Expression<T> (leftExpr.value () - rightExpr.value (),
    leftExpr.gradient () - rightExpr.gradient ());
}

template <typename TLeft, typename TRight, typename T = GetType<TLeft, TRight>::TValueType>
Expression<T> operator* (const TLeft& left, const TRight& right) {
    Expression<T> leftExpr  = OptiSik::evaluate<T, TLeft> (left);
    Expression<T> rightExpr = OptiSik::evaluate<T, TRight> (right);
    return Expression<T> (leftExpr.value () * rightExpr.value (),
    leftExpr.gradient () * rightExpr.value () +
    leftExpr.value () * rightExpr.gradient ());
}

} // namespace OptiSik