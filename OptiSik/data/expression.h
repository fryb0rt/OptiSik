#pragma once
#include <cmath>
namespace OptiSik {

//=============================================================================
//
// Class representing an expression with automatic differentiation support
//
//=============================================================================
template <typename T> class Expression {
    T mValue;
    T mGrad;

    public:
    using TValueType = T;
    Expression () : mValue (T (0)), mGrad (T (0)) {
    }

    template <typename U> Expression (U&& value) {
        *this = std::forward<U> (value);
    }

    Expression (const T& value, const T& grad) : mValue (value), mGrad (grad) {
    }

    Expression (T&& value, T&& grad)
    : mValue (std::move (value)), mGrad (std::move (grad)) {
    }

    template <typename U> Expression& operator= (U&& value) {
        if constexpr (std::is_same_v<std::decay_t<U>, Expression<T>>) {
            mValue = value.value ();
            mGrad  = value.gradient ();
        } else {
            mValue = std::forward<U> (value);
            mGrad  = T (0);
        }
        return *this;
    }

    template <typename U> operator U () const {
        return static_cast<U> (mValue);
    }

    T gradient () const {
        return mGrad;
    }
    T value () const {
        return mValue;
    }
    T& value () {
        return mValue;
    }
    void setGradient (const T& grad) {
        mGrad = grad;
    }
    Expression<T> operator- () const {
        return Expression<T> (-mValue, -mGrad);
    }
    Expression<T>& operator+ () const {
        return *this;
    }
    template <typename U> Expression<T>& operator+= (U&& other) {
        *this = *this + other;
        return *this;
    }
    template <typename U> Expression<T>& operator-= (U&& other) {
        *this = *this - other;
        return *this;
    }
    template <typename U> Expression<T>& operator*= (U&& other) {
        *this = *this * other;
        return *this;
    }
    template <typename U> Expression<T> operator/= (U&& other) {
        *this = *this / other;
        return *this;
    }
};

//=============================================================================
//
// Support for higher order derivatives
//
//=============================================================================

namespace {
template <size_t TOrder, typename T> class ExpressionHigherOrderAux {
    public:
    using Type = Expression<typename ExpressionHigherOrderAux<TOrder - 1, T>::Type>;
};

template <typename T> class ExpressionHigherOrderAux<0, T> {
    public:
    using Type = T;
};
} // namespace
template <size_t TOrder, typename T>
using ExpressionHigherOrder = typename ExpressionHigherOrderAux<TOrder, T>::Type;

template <typename T> using Expression2 = ExpressionHigherOrder<2, T>;
template <typename T> using Expression3 = ExpressionHigherOrder<3, T>;

//=============================================================================
//
// Type traits
//
//=============================================================================

template <typename T> class ExpressionInfo {
    public:
    static constexpr size_t order = 0;
    using Type                    = T;
};

template <typename T> class ExpressionInfo<Expression<T>> {
    public:
    static constexpr size_t order = ExpressionInfo<T>::order + 1;
    using Type                    = typename ExpressionInfo<T>::Type;
};

template <typename T> constexpr auto expressionValue (T&& expression) {
    if constexpr (std::is_arithmetic_v<std::decay_t<T>>)
        return expression;
    else
        return expressionValue (expression.value ());
}

namespace {
template <typename T> struct IsExpressionAux {
    static constexpr bool value = false;
};

template <typename T> struct IsExpressionAux<Expression<T>> {
    static constexpr bool value = true;
};

template <typename T> struct IsExpression {
    static constexpr bool value = IsExpressionAux<std::decay_t<T>>::value;
};

template <typename TLeft, typename TRight> struct CommonExpressionAux2 {};

template <typename T, typename TRight>
struct CommonExpressionAux2<Expression<T>, TRight> {
    using Type = Expression<T>;
};

template <typename TLeft, typename T>
struct CommonExpressionAux2<TLeft, Expression<T>> {
    using Type = Expression<T>;
};

template <typename T>
struct CommonExpressionAux2<Expression<T>, Expression<T>> {
    using Type = Expression<T>;
};

template <typename TLeft, typename TRight, typename = std::enable_if_t<IsExpression<TLeft>::value || IsExpression<TRight>::value>>
struct CommonExpression2 {
    using Type = CommonExpressionAux2<std::decay_t<TLeft>, std::decay_t<TRight>>::Type;
};

template <typename T, typename = std::enable_if_t<IsExpression<T>::value>>
struct CommonExpression {
    using Type = std::decay_t<T>;
};

} // namespace

//=============================================================================
//
// Constants
//
//=============================================================================

template <typename T> constexpr auto One () {
    return static_cast<ExpressionInfo<T>::Type> (1);
}

template <typename T> constexpr auto Zero () {
    return static_cast<ExpressionInfo<T>::Type> (0);
}

//=============================================================================
//
// Operator overloads
//
//=============================================================================

template <typename TLeft, typename TRight, typename TExpression = typename CommonExpression2<TLeft, TRight>::Type>
TExpression operator+ (TLeft&& left, TRight&& right) {
    if constexpr (!IsExpression<TLeft>::value) {
        return TExpression (left + right.value (), right.gradient ());
    } else if constexpr (!IsExpression<TRight>::value) {
        return TExpression (right + left.value (), left.gradient ());
    } else {
        return TExpression (left.value () + right.value (),
                            left.gradient () + right.gradient ());
    }
}

template <typename TLeft, typename TRight, typename TExpression = typename CommonExpression2<TLeft, TRight>::Type>
TExpression operator- (TLeft&& left, TRight&& right) {
    if constexpr (!IsExpression<TLeft>::value) {
        return TExpression (left - right.value (), right.gradient ());
    } else if constexpr (!IsExpression<TRight>::value) {
        return TExpression (right - left.value (), left.gradient ());
    } else {
        return TExpression (left.value () - right.value (),
                            left.gradient () - right.gradient ());
    }
}

template <typename TLeft, typename TRight, typename TExpression = typename CommonExpression2<TLeft, TRight>::Type>
TExpression operator* (TLeft&& left, TRight&& right) {
    if constexpr (!IsExpression<TLeft>::value) {
        return TExpression (left * right.value (), left * right.gradient ());
    } else if constexpr (!IsExpression<TRight>::value) {
        return TExpression (left.value () * right, left.gradient () * right);
    } else {
        return TExpression (left.value () * right.value (),
                            left.gradient () * right.value () +
                            left.value () * right.gradient ());
    }
}

template <typename TLeft, typename TRight, typename TExpression = typename CommonExpression2<TLeft, TRight>::Type>
TExpression operator/ (TLeft&& left, TRight&& right) {
    if constexpr (!IsExpression<TLeft>::value) {
        return TExpression (left / right.value (),
                            -(left * right.gradient ()) /
                            (right.value () * right.value ()));
    } else if constexpr (!IsExpression<TRight>::value) {
        return TExpression (left.value () / right, left.gradient () / right);
    } else {
        return TExpression (left.value () / right.value (),
                            (left.gradient () * right.value () -
                             left.value () * right.gradient ()) /
                            (right.value () * right.value ()));
    }
}

//=============================================================================
//
// Function overloads - single variable
//
//=============================================================================

using std::abs;
using std::acos;
using std::asin;
using std::atan;
using std::cos;
using std::cosh;
using std::erf;
using std::exp;
using std::log;
using std::log10;
using std::sin;
using std::sinh;
using std::sqrt;
using std::tan;
using std::tanh;

template <typename T, typename TExpression = CommonExpression<T>::Type>
TExpression abs (T&& expr) {
    auto mult = expr.value () >= Zero<TExpression> () ? One<TExpression> () :
                                                        -One<TExpression> ();
    return TExpression (abs (expr.value ()), expr.gradient () * mult);
}

template <typename T, typename TExpression = CommonExpression<T>::Type>
TExpression acos (T&& expr) {
    auto mult = -One<TExpression> () /
    sqrt (One<TExpression> () - expr.value () * expr.value ());
    return TExpression (acos (expr.value ()), expr.gradient () * mult);
}

template <typename T, typename TExpression = CommonExpression<T>::Type>
TExpression asin (T&& expr) {
    auto mult = One<TExpression> () /
    sqrt (One<TExpression> () - expr.value () * expr.value ());
    return TExpression (asin (expr.value ()), expr.gradient () * mult);
}

template <typename T, typename TExpression = CommonExpression<T>::Type>
TExpression atan (T&& expr) {
    auto mult =
    One<TExpression> () / (One<TExpression> () + expr.value () * expr.value ());
    return TExpression (atan (expr.value ()), expr.gradient () * mult);
}

template <typename T, typename TExpression = CommonExpression<T>::Type>
TExpression cos (T&& expr) {
    return TExpression (cos (expr.value ()), -expr.gradient () * sin (expr.value ()));
}

template <typename T, typename TExpression = CommonExpression<T>::Type>
TExpression cosh (T&& expr) {
    return TExpression (cosh (expr.value ()), expr.gradient () * sinh (expr.value ()));
}

template <typename T, typename TExpression = CommonExpression<T>::Type>
TExpression erf (T&& expr) {
    constexpr typename ExpressionInfo<TExpression>::Type TwoDivSqrtPi =
    2.0 / 1.7724538509055160272981674833411451872554456638435;
    return TExpression (erf (expr.value ()),
                        expr.gradient () * TwoDivSqrtPi *
                        exp (-expr.value () * expr.value ()));
}

template <typename T, typename TExpression = CommonExpression<T>::Type>
TExpression exp (T&& expr) {
    auto newValue = exp (expr.value ());
    return TExpression (newValue, expr.gradient () * newValue);
}

template <typename T, typename TExpression = CommonExpression<T>::Type>
TExpression log (T&& expr) {
    auto mult = One<TExpression> () / expr.value ();
    return TExpression (log (expr.value ()), expr.gradient () * mult);
}

template <typename T, typename TExpression = CommonExpression<T>::Type>
TExpression log10 (T&& expr) {
    constexpr typename ExpressionInfo<TExpression>::Type ln10 = 2.3025850929940456840179914546843;
    auto mult = One<TExpression> () / (expr.value () * ln10);
    return TExpression (log10 (expr.value ()), expr.gradient () * mult);
}

template <typename T, typename TExpression = CommonExpression<T>::Type>
TExpression sin (T&& expr) {
    return TExpression (sin (expr.value ()), expr.gradient () * cos (expr.value ()));
}

template <typename T, typename TExpression = CommonExpression<T>::Type>
TExpression sinh (T&& expr) {
    return TExpression (sinh (expr.value ()), expr.gradient () * cosh (expr.value ()));
}

template <typename T, typename TExpression = CommonExpression<T>::Type>
TExpression sqrt (T&& expr) {
    auto newValue = sqrt (expr.value ());
    return TExpression (newValue,
                        expr.gradient () *
                        (typename ExpressionInfo<TExpression>::Type (0.5) / newValue));
}

template <typename T, typename TExpression = CommonExpression<T>::Type>
TExpression tan (T&& expr) {
    auto cosValue = cos (expr.value ());
    auto mult     = One<TExpression> () / (cosValue * cosValue);
    return TExpression (tan (expr.value ()), expr.gradient () * mult);
}

template <typename T, typename TExpression = CommonExpression<T>::Type>
TExpression tanh (T&& expr) {
    auto cosHValue = cosh (expr.value ());
    auto mult      = One<TExpression> () / (cosHValue * cosHValue);
    return TExpression (tanh (expr.value ()), expr.gradient () * mult);
}

//=============================================================================
//
// Function overloads - two variables
//
//=============================================================================
using std::atan2;
using std::pow;

template <typename TBase, typename TExp, typename TExpression = CommonExpression2<TBase, TExp>::Type>
TExpression pow (TBase&& base, TExp&& exp) {
    if constexpr (!IsExpression<TExp>::value) {
        auto newValue = std::pow (base.value (), exp);
        auto mult = exp * std::pow (base.value (), exp - One<TExpression> ());
        return TExpression (newValue, base.gradient () * mult);
    } else if constexpr (!IsExpression<TBase>::value) {
        auto newValue = std::pow (base, exp.value ());
        auto mult     = newValue * log (base);
        return TExpression (newValue, exp.gradient () * mult);
    } else {
        auto newValue = std::pow (base.value (), exp.value ());
        auto multBase =
        exp.value () * std::pow (base.value (), exp.value () - One<TExpression> ());
        auto multExp = newValue * log (base.value ());
        return TExpression (newValue, base.gradient () * multBase + exp.gradient () * multExp);
    }
}

template <typename TNum, typename TDen, typename TExpression = CommonExpression2<TNum, TDen>::Type>
TExpression atan2 (TNum&& num, TDen&& den) {
    if constexpr (!IsExpression<TDen>::value) {
        auto denom = den * den + num.value () * num.value ();
        auto mult  = den / denom;
        return TExpression (atan2 (num.value (), den), num.gradient () * mult);
    } else if constexpr (!IsExpression<TNum>::value) {
        auto denom = den.value () * den.value () + num * num;
        auto mult  = -num / denom;
        return TExpression (atan2 (num, den.value ()), den.gradient () * mult);
    } else {
        auto denom = den.value () * den.value () + num.value () * num.value ();
        return TExpression (atan2 (num.value (), den.value ()),
                            (den.value () * num.gradient () - num.value () * den.gradient ()) / denom);
    }
}

} // namespace OptiSik