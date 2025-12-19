#include <gtest/gtest.h>
#include "data/expression.h"
#include "data/derivative.h"
#include "data/vector.h"

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

TEST(AutoDiff, powInt) {
  Expression<double> x = 4.0;
  const auto powFunc = [](Expression<double> x) {
      return pow<7>(x);
  };
  EXPECT_DOUBLE_EQ(powFunc(x), pow<7>(4.0));
  EXPECT_DOUBLE_EQ(getDerivative<1>(computeDerivative(powFunc, WithRespectTo(x), x)), 7.0 * pow<6>(4.0));
}

TEST(AutoDiff, AbsDerivative) {
  Expression<double> x = -10.0;
  const auto absFunc = [](Expression<double> x) {
      return abs(x);
  };
  EXPECT_DOUBLE_EQ(absFunc(x), 10.0);
  EXPECT_DOUBLE_EQ(getDerivative<1>(computeDerivative(absFunc, WithRespectTo(x), x)), -1.0);
  x = 10.0;
  EXPECT_DOUBLE_EQ(getDerivative<1>(computeDerivative(absFunc, WithRespectTo(x), x)), 1.0);
}

TEST(AutoDiff, ACosDerivative) {
  Expression<double> x = 0.5;
  const auto acosFunc = [](Expression<double> x) {
      return acos(x);
  };
  EXPECT_DOUBLE_EQ(acosFunc(x), std::acos(0.5));
  EXPECT_DOUBLE_EQ(getDerivative<1>(computeDerivative(acosFunc, WithRespectTo(x), x)), -1.1547005383792517);
}

TEST(AutoDiff, ASinDerivative) {
  Expression<double> x = 0.5;
  const auto asinFunc = [](Expression<double> x) {
      return asin(x);
  };
  EXPECT_DOUBLE_EQ(asinFunc(x), std::asin(0.5));
  EXPECT_DOUBLE_EQ(getDerivative<1>(computeDerivative(asinFunc, WithRespectTo(x), x)), 1.1547005383792517);
}

TEST(AutoDiff, ATanDerivative) {
  Expression<double> x = 0.5;
  const auto atanFunc = [](Expression<double> x) {
      return atan(x);
  };
  EXPECT_DOUBLE_EQ(atanFunc(x), std::atan(0.5));
  EXPECT_DOUBLE_EQ(getDerivative<1>(computeDerivative(atanFunc, WithRespectTo(x), x)), 0.8);
}

TEST(AutoDiff, CosDerivative) {
  Expression<double> x = M_PI / 4.0;
  const auto cosFunc = [](Expression<double> x) {
      return cos(x);
  };
  EXPECT_DOUBLE_EQ(cosFunc(x), std::cos(M_PI / 4.0));
  EXPECT_DOUBLE_EQ(getDerivative<1>(computeDerivative(cosFunc, WithRespectTo(x), x)), -1 / sqrt(2));
}

TEST(AutoDiff, CoshDerivative) {
  Expression<double> x = M_PI / 4.0;
  const auto coshFunc = [](Expression<double> x) {
      return cosh(x);
  };
  EXPECT_DOUBLE_EQ(coshFunc(x), std::cosh(M_PI / 4.0));
  EXPECT_DOUBLE_EQ(getDerivative<1>(computeDerivative(coshFunc, WithRespectTo(x), x)), sinh(M_PI / 4.0));
}

TEST(AutoDiff, ErfDerivative) {
  Expression<double> x = 0.5;
  const auto erfFunc = [](Expression<double> x) {
      return erf(x);
  };
  EXPECT_DOUBLE_EQ(erfFunc(x), std::erf(0.5));
  EXPECT_DOUBLE_EQ(getDerivative<1>(computeDerivative(erfFunc, WithRespectTo(x), x)), 0.87878257893544476);
}

TEST(AutoDiff, ExpDerivative) {
  Expression<double> x = 11.25;
  const auto expFunc = [](Expression<double> x) {
      return exp(x);
  };
  EXPECT_DOUBLE_EQ(expFunc(x), std::exp(11.25));
  EXPECT_DOUBLE_EQ(getDerivative<1>(computeDerivative(expFunc, WithRespectTo(x), x)), exp(11.25));
}

TEST(AutoDiff, LogDerivative) {
  Expression<double> x = 11.25;
  const auto logFunc = [](Expression<double> x) {
      return log(x);
  };
  EXPECT_DOUBLE_EQ(logFunc(x), std::log(11.25));
  EXPECT_DOUBLE_EQ(getDerivative<1>(computeDerivative(logFunc, WithRespectTo(x), x)), 1.0 / 11.25);
}

TEST(AutoDiff, Log10Derivative) {
  Expression<double> x = 11.25;
  const auto log10Func = [](Expression<double> x) {
      return log10(x);
  };
  EXPECT_DOUBLE_EQ(log10Func(x), std::log10(11.25));
  EXPECT_DOUBLE_EQ(getDerivative<1>(computeDerivative(log10Func, WithRespectTo(x), x)), 1.0 / (x * log(10)));
}

TEST(AutoDiff, SinDerivative) {
  Expression<double> x = M_PI / 4.0;
  const auto sinFunc = [](Expression<double> x) {
      return sin(x);
  };
  EXPECT_DOUBLE_EQ(sinFunc(x), std::sin(M_PI / 4.0));
  EXPECT_DOUBLE_EQ(getDerivative<1>(computeDerivative(sinFunc, WithRespectTo(x), x)), 1 / sqrt(2));
}

TEST(AutoDiff, SinhDerivative) {
  Expression<double> x = M_PI / 4.0;
  const auto sinhFunc = [](Expression<double> x) {
      return sinh(x);
  };
  EXPECT_DOUBLE_EQ(sinhFunc(x), std::sinh(M_PI / 4.0));
  EXPECT_DOUBLE_EQ(getDerivative<1>(computeDerivative(sinhFunc, WithRespectTo(x), x)), cosh(M_PI / 4.0));
}

TEST(AutoDiff, sqrt) {
  Expression<double> x = M_PI / 4.0;
  const auto sqrtFunc = [](Expression<double> x) {
      return sqrt(x);
  };
  EXPECT_DOUBLE_EQ(sqrtFunc(x), std::sqrt(M_PI / 4.0));
  EXPECT_DOUBLE_EQ(getDerivative<1>(computeDerivative(sqrtFunc, WithRespectTo(x), x)), 1.0 / (2.0 * sqrt(M_PI / 4.0)));
}

TEST(AutoDiff, tan) {
  Expression<double> x = M_PI / 4.0;
  const auto tanFunc = [](Expression<double> x) {
      return tan(x);
  };
  EXPECT_DOUBLE_EQ(tanFunc(x), std::tan(M_PI / 4.0));
  EXPECT_DOUBLE_EQ(getDerivative<1>(computeDerivative(tanFunc, WithRespectTo(x), x)), 2.0);
}

TEST(AutoDiff, tanh) {
  Expression<double> x = M_PI / 4.0;
  const auto tanhFunc = [](Expression<double> x) {
      return tanh(x);
  };
  EXPECT_DOUBLE_EQ(tanhFunc(x), std::tanh(M_PI / 4.0));
  EXPECT_DOUBLE_EQ(getDerivative<1>(computeDerivative(tanhFunc, WithRespectTo(x), x)), 1.0 / cosh(M_PI / 4.0) / cosh(M_PI / 4.0));
}

TEST(AutoDiff, pow) {
  Expression<double> x = 4.0;
  Expression<double> y = 3.0;
  const auto powFuncEE = [](Expression<double> x, Expression<double> y) {
      return pow(x, y);
  };
  EXPECT_DOUBLE_EQ(powFuncEE(x, y), std::pow(4.0, 3.0));
  EXPECT_DOUBLE_EQ(getDerivative<1>(computeDerivative(powFuncEE, WithRespectTo(x), x, y)), 3.0 * std::pow(4.0, 2.0));
  EXPECT_DOUBLE_EQ(getDerivative<1>(computeDerivative(powFuncEE, WithRespectTo(y), x, y)), std::pow(4.0, 3.0) * log(4.0));

  const auto powFuncEV = [](Expression<double> x, const double y) {
      return pow(x, y);
  };
  EXPECT_DOUBLE_EQ(getDerivative<1>(computeDerivative(powFuncEV, WithRespectTo(x), x, 3.0)), 3.0 * std::pow(4.0, 2.0));
  
  const auto powFuncVE = [](const double x, Expression<double> y) {
      return pow(x, y);
  };
  EXPECT_DOUBLE_EQ(getDerivative<1>(computeDerivative(powFuncVE, WithRespectTo(y), 4.0, y)), std::pow(4.0, 3.0) * log(4.0));
}

TEST(AutoDiff, atan2) {
  Expression<double> x = 4.0;
  Expression<double> y = 3.0;
  const auto atan2FuncEE = [](Expression<double> x, Expression<double> y) {
      return atan2(x, y);
  };
  EXPECT_DOUBLE_EQ(atan2FuncEE(x, y), std::atan2(4.0, 3.0));
  EXPECT_DOUBLE_EQ(getDerivative<1>(computeDerivative(atan2FuncEE, WithRespectTo(x), x, y)), 0.12);
  EXPECT_DOUBLE_EQ(getDerivative<1>(computeDerivative(atan2FuncEE, WithRespectTo(y), x, y)), -0.16);

  const auto atan2FuncEV = [](Expression<double> x, const double y) {
      return atan2(x, y);
  };
  EXPECT_DOUBLE_EQ(getDerivative<1>(computeDerivative(atan2FuncEV, WithRespectTo(x), x, 3.0)), 0.12);
  
  const auto atan2FuncVE = [](const double x, Expression<double> y) {
      return atan2(x, y);
  };
  EXPECT_DOUBLE_EQ(getDerivative<1>(computeDerivative(atan2FuncVE, WithRespectTo(y), 4.0, y)), -0.16);
}

TEST(AutoDiff, HigherOrderPow) {
  using Exp3 = Expression3<double>;
  Exp3 a = 3.0;
  Exp3 b = 4.0;
  const auto func = [](Exp3 a, Exp3 b) {
      return pow(a,b);
  };
  auto res = computeDerivative(func, WithRespectTo(a,a,a), a, b);
  EXPECT_DOUBLE_EQ(getDerivative<0>(res), 81.0);
  EXPECT_DOUBLE_EQ(getDerivative<1>(res), 108.0);
  EXPECT_DOUBLE_EQ(getDerivative<2>(res), 108.0);
  EXPECT_DOUBLE_EQ(getDerivative<3>(res), 72.0);
}

TEST(AutoDiff, HigherOrderPow2) {
  using Exp3 = Expression3<double>;
  Exp3 a = 3.0;
  Exp3 b = 4.0;
  const auto func = [](Exp3 a, Exp3 b) {
      return pow(a,b);
  };
  auto res = computeDerivative(func, WithRespectTo(b,b,b), a, b);
  EXPECT_DOUBLE_EQ(getDerivative<0>(res), 81.0);
  EXPECT_DOUBLE_EQ(getDerivative<1>(res), 81.0 * log(3));
  EXPECT_DOUBLE_EQ(getDerivative<2>(res), 81.0 * pow<2>(log(3)));
  EXPECT_DOUBLE_EQ(getDerivative<3>(res), 81.0 * pow<3>(log(3)));
}

TEST(AutoDiff, HigherOrderPow3) {
  using Exp3 = Expression3<double>;
  Exp3 a = 3.0;
  Exp3 b = 4.0;
  const auto func = [](Exp3 a, Exp3 b) {
      return pow(a,b);
  };
  auto res = computeDerivative(func, WithRespectTo(a, b, a), a, b);
  EXPECT_DOUBLE_EQ(getDerivative<0>(res), 81.0);
  EXPECT_DOUBLE_EQ(getDerivative<1>(res), 108);
  EXPECT_DOUBLE_EQ(getDerivative<2>(res), 108 * log(3) + 27.0);
  EXPECT_DOUBLE_EQ(getDerivative<3>(res), 108 * log(3) + 63.0);
}

TEST(AutoDiff, HigherOrderFunctions) {
  using Exp3 = Expression3<double>;
  Exp3 a = 3.0;
  Exp3 b = 4.0;
  Exp3 c = 5.0;
  const auto func = [](Exp3 a, Exp3 b, Exp3 c) {
      return abs(sin(a + exp(b)) * log(c) + pow(a,b) * atan2(b, c) / sinh(a));
  };
  auto res = computeDerivative(func, WithRespectTo(a, b, c), a, b, c);
  EXPECT_DOUBLE_EQ(getDerivative<0>(res), 6.8512987317001866);
  EXPECT_DOUBLE_EQ(getDerivative<1>(res), 2.5929686716269345);
  EXPECT_DOUBLE_EQ(getDerivative<2>(res), -72.089431210219814);
  EXPECT_DOUBLE_EQ(getDerivative<3>(res), -10.030853028022817);
}

TEST(AutoDiff, DerivativeResultVector) {
  using Exp3 = Expression3<double>;
  Exp3 a = 3.0;
  Exp3 b = 4.0;
  Exp3 c = 5.0;
  const auto func = [](Exp3 a, Exp3 b, Exp3 c) {
      return abs(sin(a + exp(b)) * log(c) + pow(a,b) * atan2(b, c) / sinh(a));
  };
  SVector<double, 3> result = derivative(func, WithRespectTo(a, b, c), a, b, c);
  EXPECT_DOUBLE_EQ(result[0], 2.5929686716269345);
  EXPECT_DOUBLE_EQ(result[1], -72.089431210219814);
  EXPECT_DOUBLE_EQ(result[2], -10.030853028022817);
}

TEST(AutoDiff, Gradient) {
   using Exp3 = Expression3<double>;
  Exp3 a = 3.0;
  Exp3 b = 4.0;
  Exp3 c = 5.0;
  const auto func = [](Exp3 a, Exp3 b, Exp3 c) {
      return abs(sin(a + exp(b)) * log(c) + pow(a,b) * atan2(b, c) / sinh(a));
  };
  SVector<double, 3> result = gradient(func, a, b, c);
  EXPECT_DOUBLE_EQ(result[0], 2.5929686716269345);
  EXPECT_DOUBLE_EQ(result[1], -72.089431210219814);
  EXPECT_DOUBLE_EQ(result[2], -10.030853028022817);
}