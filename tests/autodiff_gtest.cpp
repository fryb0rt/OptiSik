#include <gtest/gtest.h>
#include "data/expression.h"
#include "data/derivative.h"

using namespace OptiSik;

TEST(AutoDiff, PlusDerivative) {
  Expression<double> a = 3.0;
  const auto func = [](Expression<double> x) {
      return 3.0 + (x + 4.0) - 5 + x + x - x;
  };
  const double deriv = getDerivative<1>(computeDerivative(func, WithRespectTo(a), a));
  EXPECT_DOUBLE_EQ(deriv, 2.0);
}

TEST(AutoDiff, MultDerivative) {
  Expression<double> a = 3.0;
  const auto func = [](Expression<double> x) {
      return 4.0 * x * x;
  };
  const double deriv = getDerivative<1>(computeDerivative(func, WithRespectTo(a), a));
  EXPECT_DOUBLE_EQ(deriv, 24.0);
}

TEST(AutoDiff, HigherOrderDerivative) {
  using Exp2 = Expression<Expression<double>>;
  Exp2 a = 3.0;
  const auto func = [](Exp2 x) {
      return x * x * x + x * x + x;
  };
  auto res = computeDerivative(func, WithRespectTo(a, a), a);
  EXPECT_DOUBLE_EQ(getDerivative<0>(res), 39.0);
  EXPECT_DOUBLE_EQ(getDerivative<1>(res), 34.0);
  EXPECT_DOUBLE_EQ(getDerivative<2>(res), 20.0);
}

TEST(AutoDiff, HigherOrderDerivative2) {
  using Exp3 = Expression<Expression<Expression<double>>>;
  Exp3 a = 3.0;
  const auto func = [](Exp3 x) {
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
  using Exp2 = Expression<Expression<double>>;
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