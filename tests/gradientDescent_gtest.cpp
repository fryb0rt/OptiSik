#include "data/derivative.h"
#include "data/vector.h"
#include "optimizers/gradientDescent.h"
#include <gtest/gtest.h>



using namespace OptiSik;

TEST(GradientDescentTest, SimpleQuadratic) {
    // Minimize f(x) = x^2, gradient = 2x
    auto objective = [](const Vector<double>& x) { return x[0] * x[0]; };
    auto gradient  = [](const Vector<double>& x) {
        Vector<double> grad(1);
        grad[0] = 2.0 * x[0];
        return grad;
    };

    Vector<double> x0(1);
    x0[0] = 5.0;

    GradientDescent<double>::Config config;
    config.learningRate = 0.1;
    config.tolerance    = 1e-6;

    auto result = GradientDescent<double>::minimize(x0, objective, gradient, config);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.x[0], 0.0, 1e-4);
    EXPECT_NEAR(result.value, 0.0, 1e-6);
}

TEST(GradientDescentTest, Rosenbrock1D) {
    // Simplified Rosenbrock in 1D: f(x) = (1-x)^2
    // Minimum at x=1
    auto objective = [](const Vector<double>& x) {
        double diff = 1.0 - x[0];
        return diff * diff;
    };
    auto gradient = [](const Vector<double>& x) {
        Vector<double> grad(1);
        grad[0] = -2.0 * (1.0 - x[0]);
        return grad;
    };

    Vector<double> x0(1);
    x0[0] = 0.0;

    GradientDescent<double>::Config config;
    config.learningRate = 0.1;
    config.tolerance    = 1e-6;

    auto result = GradientDescent<double>::minimize(x0, objective, gradient, config);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.x[0], 1.0, 1e-4);
}

TEST(GradientDescentTest, Multivariate) {
    // Minimize f(x,y) = (x-2)^2 + (y-3)^2
    // Minimum at (2, 3) with value 0
    auto objective = [](const Vector<double>& x) {
        double dx = x[0] - 2.0;
        double dy = x[1] - 3.0;
        return dx * dx + dy * dy;
    };
    auto gradient = [](const Vector<double>& x) {
        Vector<double> grad(2);
        grad[0] = 2.0 * (x[0] - 2.0);
        grad[1] = 2.0 * (x[1] - 3.0);
        return grad;
    };

    Vector<double> x0(2);
    x0[0] = 0.0;
    x0[1] = 0.0;

    GradientDescent<double>::Config config;
    config.learningRate = 0.1;
    config.tolerance    = 1e-6;

    auto result = GradientDescent<double>::minimize(x0, objective, gradient, config);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.x[0], 2.0, 1e-4);
    EXPECT_NEAR(result.x[1], 3.0, 1e-4);
    EXPECT_NEAR(result.value, 0.0, 1e-6);
}

TEST(GradientDescentTest, MultivariateAuto) {
    // Minimize f(x,y) = (x-2)^2 + (y-3)^2
    // Minimum at (2, 3) with value 0
    auto objective = [](Expression<double> x, Expression<double> y) {
        auto dx = x - 2.0;
        auto dy = y - 3.0;
        return dx * dx + dy * dy;
    };

    SVector<double, 2> x0;
    x0[0] = 0.0;
    x0[1] = 0.0;

    GradientDescent<double>::Config config;
    config.learningRate = 0.1;
    config.tolerance    = 1e-6;

    auto result = GradientDescent<double>::minimize(
    x0,
    [&objective](const SVector<double, 2>& x) {
        return static_cast<double>(objective(x[0], x[1]));
    },
    [&objective](const SVector<double, 2>& x) {
        return gradient(objective,Expression<double>(x[0]), Expression<double>(x[1]));
    }, config);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.x[0], 2.0, 1e-4);
    EXPECT_NEAR(result.x[1], 3.0, 1e-4);
    EXPECT_NEAR(result.value, 0.0, 1e-6);
}
