#include "data/vector.h"
#include <gtest/gtest.h>

using namespace OptiSik;
using Vec = Vector<double>;

TEST(VectorTest, List) {
    Vec v1({1,2,3});
    EXPECT_DOUBLE_EQ(v1[0], 1.0);
    EXPECT_DOUBLE_EQ(v1[1], 2.0);
    EXPECT_DOUBLE_EQ(v1[2], 3.0);
}

TEST(VectorTest, DotAndScalar) {
    Vec v1(std::vector<double>{ 1, 2, 3 });
    Vec v2(std::vector<double>{ 4, 5, 6 });
    EXPECT_DOUBLE_EQ(v1.dot(v2), 32.0);
    Vec v3 = v1 * 2.0;
    EXPECT_DOUBLE_EQ(v3[0], 2.0);
    v3 *= 0.5;
    EXPECT_TRUE(v3 == v1);
}

TEST(VectorTest, Arithmetic) {
    Vec a(std::vector<double>{ 5, 6 });
    Vec b(std::vector<double>{ 2, 8 });
    EXPECT_EQ((a + b), Vec(std::vector<double>{ 7, 14 }));
    EXPECT_EQ((a - b), Vec(std::vector<double>{ 3, -2 }));
    a -= b;
    EXPECT_EQ(a, Vec(std::vector<double>{ 3, -2 }));
    EXPECT_EQ(2.0 * b, Vec(std::vector<double>{ 4, 16 }));
}

TEST(VectorTest, MinMaxAndStats) {
    Vec x(std::vector<double>{ 1, 9, 3 });
    Vec y(std::vector<double>{ 4, 2, 8 });
    EXPECT_EQ(min(x, y), Vec(std::vector<double>{ 1, 2, 3 }));
    EXPECT_EQ(max(x, y), Vec(std::vector<double>{ 4, 9, 8 }));
    EXPECT_DOUBLE_EQ(x.minElement(), 1.0);
    EXPECT_DOUBLE_EQ(x.maxElement(), 9.0);
    EXPECT_EQ(x.minArg(), 0u);
    EXPECT_EQ(x.maxArg(), 1u);
}

TEST(VectorTest, NormalizeAverageIterators) {
    Vec v(std::vector<double>{ 3, 4 });
    Vec n = v.normalized();
    EXPECT_NEAR(n.magnitude(), 1.0, 1e-8);
    Vec z(std::vector<double>{ 1, 2, 3, 4 });
    EXPECT_DOUBLE_EQ(z.average(), 2.5);
    Vec f(4);
    f.fill(7.0);
    double s = 0;
    for (auto& x : f)
        s += x;
    EXPECT_DOUBLE_EQ(s, 28.0);
}

TEST(VectorTest, EpsilonEquals) {
    Vec a(std::vector<double>{ 1.0, 2.0 });
    Vec b(std::vector<double>{ 1.001, 1.999 });
    EXPECT_TRUE(a.epsilonEquals(b, 0.01));
    EXPECT_FALSE(a.epsilonEquals(b, 0.0001));
}

TEST(VectorTest, DifferentTypes) {
    Vector<double> a({ 1.0, 2.0, 3.0 });
    SVector<double, 3> b({ 1.0, 2.0, 3.0 });

    EXPECT_DOUBLE_EQ(a.dot(a), 14.0);
    a = b * 2.0;
    EXPECT_DOUBLE_EQ(a[0], 2.0);
    a *= 0.5;
    EXPECT_TRUE(a == b);

    EXPECT_TRUE(a.epsilonEquals(b, 0.001));

    EXPECT_EQ((a + b), Vec(std::vector<double>{ 2.0, 4.0, 6.0 }));
    a += SVector<double, 3>({ 1.0, 7.0, -2.0 });
    EXPECT_EQ((a - b), Vec(std::vector<double>{ 1.0, 7.0, -2.0 }));
    a -= b;
    EXPECT_EQ(a, Vec(std::vector<double>{ 1.0, 7.0, -2.0 }));
    EXPECT_EQ(2.0 * b, Vec(std::vector<double>{ 2.0, 4.0, 6.0 }));

    EXPECT_EQ(min(a, b), Vec(std::vector<double>{ 1.0, 2.0, -2.0 }));
    EXPECT_EQ(max(b, a), Vec(std::vector<double>{ 1.0, 7.0, 3.0 }));
    EXPECT_DOUBLE_EQ(a.minElement(), -2.0);
    EXPECT_DOUBLE_EQ(b.maxElement(), 3.0);
    EXPECT_EQ(a.minArg(), 2u);
    EXPECT_EQ(b.maxArg(), 2u);

    SVector<double, 2> v({ 3, 4 });
    SVector<double, 2> n = v.normalized();
    EXPECT_NEAR(n.magnitude(), 1.0, 1e-8);
    SVector<double, 4> z({ 1, 2, 3, 4 });
    EXPECT_DOUBLE_EQ(z.average(), 2.5);
    SVector<double, 4> f;
    f.fill(7.0);
    double s = 0;
    for (auto& x : f)
        s += x;
    EXPECT_DOUBLE_EQ(s, 28.0);
}

TEST(VectorTest, DifferentTypes2) {
    Vector<double> b({ 1.0, 2.0, 3.0 });
    SVector<double, 3> a({ 1.0, 2.0, 3.0 });

    EXPECT_DOUBLE_EQ(a.dot(a), 14.0);
    a = b * 2.0;
    EXPECT_DOUBLE_EQ(a[0], 2.0);
    a *= 0.5;
    EXPECT_TRUE(a == b);

    EXPECT_TRUE(a.epsilonEquals(b, 0.001));

    EXPECT_EQ((a + b), Vec(std::vector<double>{ 2.0, 4.0, 6.0 }));
    a += SVector<double, 3>({ 1.0, 7.0, -2.0 });
    EXPECT_EQ((a - b), Vec(std::vector<double>{ 1.0, 7.0, -2.0 }));
    a -= b;
    EXPECT_EQ(a, Vec(std::vector<double>{ 1.0, 7.0, -2.0 }));
    EXPECT_EQ(2.0 * b, Vec(std::vector<double>{ 2.0, 4.0, 6.0 }));

    EXPECT_EQ(min(a, b), Vec(std::vector<double>{ 1.0, 2.0, -2.0 }));
    EXPECT_EQ(max(b, a), Vec(std::vector<double>{ 1.0, 7.0, 3.0 }));
    EXPECT_DOUBLE_EQ(a.minElement(), -2.0);
    EXPECT_DOUBLE_EQ(b.maxElement(), 3.0);
    EXPECT_EQ(a.minArg(), 2u);
    EXPECT_EQ(b.maxArg(), 2u);

    SVector<double, 2> v({ 3, 4 });
    SVector<double, 2> n = v.normalized();
    EXPECT_NEAR(n.magnitude(), 1.0, 1e-8);
    SVector<double, 4> z({ 1, 2, 3, 4 });
    EXPECT_DOUBLE_EQ(z.average(), 2.5);
    SVector<double, 4> f;
    f.fill(7.0);
    double s = 0;
    for (auto& x : f)
        s += x;
    EXPECT_DOUBLE_EQ(s, 28.0);
}

TEST(VectorTest, StaticVector) {
    SVector<double, 3> b({ 1.0, 2.0, 3.0 });
    SVector<double, 3> a({ 1.0, 2.0, 3.0 });

    EXPECT_DOUBLE_EQ(a.dot(a), 14.0);
    a = b * 2.0;
    EXPECT_DOUBLE_EQ(a[0], 2.0);
    a *= 0.5;
    EXPECT_TRUE(a == b);

    EXPECT_TRUE(a.epsilonEquals(b, 0.001));

    EXPECT_EQ((a + b), Vec(std::vector<double>{ 2.0, 4.0, 6.0 }));
    a += SVector<double, 3>({ 1.0, 7.0, -2.0 });
    EXPECT_EQ((a - b), Vec(std::vector<double>{ 1.0, 7.0, -2.0 }));
    a -= b;
    EXPECT_EQ(a, Vec(std::vector<double>{ 1.0, 7.0, -2.0 }));
    EXPECT_EQ(2.0 * b, Vec(std::vector<double>{ 2.0, 4.0, 6.0 }));

    EXPECT_EQ(min(a, b), Vec(std::vector<double>{ 1.0, 2.0, -2.0 }));
    EXPECT_EQ(max(b, a), Vec(std::vector<double>{ 1.0, 7.0, 3.0 }));
    EXPECT_DOUBLE_EQ(a.minElement(), -2.0);
    EXPECT_DOUBLE_EQ(b.maxElement(), 3.0);
    EXPECT_EQ(a.minArg(), 2u);
    EXPECT_EQ(b.maxArg(), 2u);

    SVector<double, 2> v({ 3, 4 });
    SVector<double, 2> n = v.normalized();
    EXPECT_NEAR(n.magnitude(), 1.0, 1e-8);
    SVector<double, 4> z({ 1, 2, 3, 4 });
    EXPECT_DOUBLE_EQ(z.average(), 2.5);
    SVector<double, 4> f;
    f.fill(7.0);
    double s = 0;
    for (auto& x : f)
        s += x;
    EXPECT_DOUBLE_EQ(s, 28.0);
}

TEST(VectorTest, DifferentTypesConstructors) {
    SVector<double, 3> s({ 1.0, 2.0, 3.0 });
    EXPECT_TRUE(Vector<double>(s) == s);

    Vector<double> d({ 1.0, 2.0, 3.0 });
    SVector<double, 3> s2(d);
    EXPECT_TRUE(s2 == d);
}