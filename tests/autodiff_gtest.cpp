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