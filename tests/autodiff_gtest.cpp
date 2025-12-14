#include <gtest/gtest.h>
#include "data/expression.h"

using namespace OptiSik;

TEST(AutoDiff, Plus) {
  Expression<double> a = 3.0;
  auto expr = (a + 4.0) - 5;
  EXPECT_DOUBLE_EQ(expr.evaluate(), 2.0);
}