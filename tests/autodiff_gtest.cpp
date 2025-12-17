#include <gtest/gtest.h>
#include "data/expression.h"
#include "data/derivative.h"

using namespace OptiSik;

TEST(AutoDiff, PlusMinusDerivative) {
  Expression<double> a = 3.0;
  const auto func = [](Expression<double> x) {
      return 3.0 + (x + 4.0) - 5 + x + x - x;
  };
  const double deriv = getDerivative<1>(computeDerivative(func, WithRespectTo(a), a));
  EXPECT_DOUBLE_EQ(deriv, 2.0);
}

TEST(AutoDiff, UnaryMinusDerivative) {
  Expression<double> a = 3.0;
  const auto func = [](Expression<double> x) {
      return -x;
  };
  const double deriv = getDerivative<1>(computeDerivative(func, WithRespectTo(a), a));
  EXPECT_DOUBLE_EQ(deriv, -1.0);
}

TEST(AutoDiff, MultDerivative) {
  Expression<double> a = 3.0;
  const auto func = [](Expression<double> x) {
      return 4.0 * x * x;
  };
  const double deriv = getDerivative<1>(computeDerivative(func, WithRespectTo(a), a));
  EXPECT_DOUBLE_EQ(deriv, 24.0);
}

TEST(AutoDiff, DivDerivative) {
  Expression<double> a = 3.0;
  const auto func = [](Expression<double> x) {
      return (x + x) / (x * x);
  };
  const double deriv = getDerivative<1>(computeDerivative(func, WithRespectTo(a), a));
  EXPECT_DOUBLE_EQ(deriv, -2.0 / 9.0);
}

TEST(AutoDiff, HigherOrderDerivative) {
  Expression2<double> a = 3.0;
  const auto func = [](Expression2<double> x) {
      return x * x * x + x * x + x;
  };
  auto res = computeDerivative(func, WithRespectTo(a, a), a);
  EXPECT_DOUBLE_EQ(getDerivative<0>(res), 39.0);
  EXPECT_DOUBLE_EQ(getDerivative<1>(res), 34.0);
  EXPECT_DOUBLE_EQ(getDerivative<2>(res), 20.0);
}

TEST(AutoDiff, HigherOrderDerivative2) {
  Expression3<double> a = 3.0;
  const auto func = [](Expression3<double> x) {
      return x * x * x + x * x + x;
  };
  auto res = computeDerivative(func, WithRespectTo(a, a, a), a);
  EXPECT_DOUBLE_EQ(a, 3.0);
  EXPECT_DOUBLE_EQ(res, 39.0);
  EXPECT_DOUBLE_EQ(getDerivative<0>(res), 39.0);
  EXPECT_DOUBLE_EQ(getDerivative<1>(res), 34.0);
  EXPECT_DOUBLE_EQ(getDerivative<2>(res), 20.0);
  EXPECT_DOUBLE_EQ(getDerivative<3>(res), 6.0);
}

TEST(AutoDiff, HigherOrderDerivative3) {
  using Exp2 = Expression2<double>;
  Exp2 a = 3.0;
  Exp2 b = 4.0;
  const auto func = [](Exp2 a, Exp2 b) {
      return a * a + a * b + b * b + a + b;
  };
  auto res = computeDerivative(func, WithRespectTo(a, b), a, b);
  EXPECT_DOUBLE_EQ(getDerivative<0>(res), 44.0);
  EXPECT_DOUBLE_EQ(getDerivative<1>(res), 11.0);
  EXPECT_DOUBLE_EQ(getDerivative<2>(res), 1.0);
}

TEST(AutoDiff, AssignDerivative) {
  Expression2<double> a = 3.0;
  const auto func = [](Expression2<double> x) {
      auto y = x;
      y -= x * 0.5;
      y *= 3.0;
      y *= y;
      y /= 2.0;
      return y;
  };
  auto res = computeDerivative(func, WithRespectTo(a, a), a);
  EXPECT_DOUBLE_EQ(getDerivative<1>(res), 6.75);
  EXPECT_DOUBLE_EQ(getDerivative<2>(res), 2.25);
}

TEST(AutoDiff, AbsDerivative) {
  Expression<double> x = -10.0;
  const auto absFunc = [](Expression<double> x) {
      return abs(x);
  };
  EXPECT_DOUBLE_EQ(getDerivative<1>(computeDerivative(absFunc, WithRespectTo(x), x)), -1.0);
  x = 10.0;
  EXPECT_DOUBLE_EQ(getDerivative<1>(computeDerivative(absFunc, WithRespectTo(x), x)), 1.0);
}

TEST(AutoDiff, ACosDerivative) {
  Expression<double> x = 0.5;
  const auto acosFunc = [](Expression<double> x) {
      return acos(x);
  };
  EXPECT_DOUBLE_EQ(getDerivative<1>(computeDerivative(acosFunc, WithRespectTo(x), x)), -1.1547005383792517);
}

TEST(AutoDiff, ASinDerivative) {
  Expression<double> x = 0.5;
  const auto asinFunc = [](Expression<double> x) {
      return asin(x);
  };
  EXPECT_DOUBLE_EQ(getDerivative<1>(computeDerivative(asinFunc, WithRespectTo(x), x)), 1.1547005383792517);
}

TEST(AutoDiff, ATanDerivative) {
  Expression<double> x = 0.5;
  const auto atanFunc = [](Expression<double> x) {
      return atan(x);
  };
  EXPECT_DOUBLE_EQ(getDerivative<1>(computeDerivative(atanFunc, WithRespectTo(x), x)), 0.8);
}

TEST(AutoDiff, SinDerivative) {
  Expression<double> x = M_PI / 4.0;
  const auto sinFunc = [](Expression<double> x) {
      return sin(x);
  };
  EXPECT_DOUBLE_EQ(getDerivative<1>(computeDerivative(sinFunc, WithRespectTo(x), x)), 1 / sqrt(2));
}