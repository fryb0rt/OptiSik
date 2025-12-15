#include <gtest/gtest.h>
#include "data/expression.h"
#include "data/derivative.h"

using namespace OptiSik;

TEST(AutoDiff, PlusDerivative) {
  Expression<double> a = 3.0;
  const auto func = [](Expression<double> x) {
      return 3.0 + (x + 4.0) - 5 + x + x - x;
  };
  const double deriv = derivative<double>(func, a, a);
  EXPECT_DOUBLE_EQ(deriv, 2.0);
}

TEST(AutoDiff, MultDerivative) {
  Expression<double> a = 3.0;
  const auto func = [](Expression<double> x) {
      return 4.0 * x * x;
  };
  const double deriv = derivative<double>(func, a, a);
  EXPECT_DOUBLE_EQ(deriv, 24.0);
}

TEST(AutoDiff, HigherOrderDerivative) {
  using Exp2 = Expression<Expression<double>>;
  Exp2 a(3.0);
  const auto func = [](Exp2 x) {
      return 4.0 * x * x;
  };
  a.setGradient(Expression<double>(1.0, 1.0));
  auto res = func(a);
  EXPECT_DOUBLE_EQ(res.gradient().value(), 24.0);
  EXPECT_DOUBLE_EQ(res.gradient().gradient(), 8.0);
}