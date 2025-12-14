#pragma once

namespace OptiSik {
template <typename T, typename TInput> T evaluate(const TInput& input) {
    if constexpr (std::is_arithmetic_v<TInput>)
        return static_cast<T> (input);
    else
        return input.evaluate ();
}

template <typename TInput> class UnaryExpression {
    protected:
    TInput mInput;

    public:
    using TValueType = typename TInput::TValueType;
    explicit UnaryExpression (const TInput& input) : mInput (input) {
    }
};

template <typename TOperand, typename TLeft, typename TRight>
class BinaryExpression {
    protected:
    TLeft mLeft;
    TRight mRight;

    public:
    using TValueType = typename TLeft::TValueType;
    BinaryExpression (const TLeft& left, const TRight& right)
    : mLeft (left), mRight (right) {
    }
    TValueType evaluate () const {
        return TOperand::template evaluate<TValueType,TLeft,TRight>(this->mLeft, this->mRight);
    }
}; // namespace OptiSik

struct PlusExpression {
    template <typename T, typename TLeft, typename TRight>
    static T evaluate (const TLeft& left, const TRight& right) {
        return OptiSik::evaluate<T, TLeft> (left) + OptiSik::evaluate<T, TRight> (right);
    }
};

template <typename TLeft, typename TRight>
BinaryExpression<PlusExpression, TLeft, TRight>
operator+(const TLeft& left, const TRight& right) {
    if constexpr (std::is_arithmetic_v<TLeft>) {
        return BinaryExpression<PlusExpression, TRight, TLeft> (left, right);
    } else {
        return BinaryExpression<PlusExpression, TLeft, TRight> (left, right);
    }
}

struct MinusExpression {
    template <typename T, typename TLeft, typename TRight>
    static T evaluate (const TLeft& left, const TRight& right) {
        return OptiSik::evaluate<T, TLeft> (left) - OptiSik::evaluate<T, TRight> (right);
    }
};

template <typename TLeft, typename TRight>
BinaryExpression<MinusExpression, TLeft, TRight>
operator-(const TLeft& left, const TRight& right) {
    if constexpr (std::is_arithmetic_v<TLeft>) {
        return BinaryExpression<MinusExpression, TRight, TLeft> (left, right);
    } else {
        return BinaryExpression<MinusExpression, TLeft, TRight> (left, right);
    }
}

template <typename T> class Expression {
    T mValue;
    T mGrad;
    public:
    using TValueType = T;
    Expression(const T& value) : mValue(value), mGrad(T(0)) {
    }
    Expression(const T& value, const T& grad) : mValue(value), mGrad(grad) {
    }
    T operator T() const {
        return mValue;
    }
    T grad() const {
        return mGrad;
    }
};

template<typename T, typename TFunction>
} // namespace OptiSik