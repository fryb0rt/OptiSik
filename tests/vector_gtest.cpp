#include <gtest/gtest.h>
#include "../data/vector.h"

using namespace OptiSik;
using Vec = Vector<double>;

TEST(VectorTest, DotAndScalar) {
  Vec v1(std::vector<double>{1,2,3});
  Vec v2(std::vector<double>{4,5,6});
  EXPECT_DOUBLE_EQ(v1.dot(v2), 32.0);
  Vec v3 = v1 * 2.0;
  EXPECT_DOUBLE_EQ(v3[0], 2.0);
  v3 *= 0.5;
  EXPECT_TRUE(v3 == v1);
}

TEST(VectorTest, Arithmetic) {
  Vec a(std::vector<double>{5,6});
  Vec b(std::vector<double>{2,8});
  EXPECT_EQ((a+b), Vec(std::vector<double>{7,14}));
  EXPECT_EQ((a-b), Vec(std::vector<double>{3,-2}));
  a -= b;
  EXPECT_EQ(a, Vec(std::vector<double>{3,-2}));
  EXPECT_EQ(2.0 * b, Vec(std::vector<double>{4,16}));
}

TEST(VectorTest, MinMaxAndStats) {
  Vec x(std::vector<double>{1,9,3});
  Vec y(std::vector<double>{4,2,8});
  EXPECT_EQ(x.min(y), Vec(std::vector<double>{1,2,3}));
  EXPECT_EQ(x.max(y), Vec(std::vector<double>{4,9,8}));
  EXPECT_DOUBLE_EQ(x.minElement(), 1.0);
  EXPECT_DOUBLE_EQ(x.maxElement(), 9.0);
  EXPECT_EQ(x.minArg(), 0u);
  EXPECT_EQ(x.maxArg(), 1u);
}

TEST(VectorTest, NormalizeAverageIterators) {
  Vec v(std::vector<double>{3,4});
  Vec n = v.normalized();
  EXPECT_NEAR(n.magnitude(), 1.0, 1e-8);
  Vec z(std::vector<double>{1,2,3,4});
  EXPECT_DOUBLE_EQ(z.average(), 2.5);
  Vec f(4);
  f.fill(7.0);
  double s = 0;
  for (auto &x : f) s += x;
  EXPECT_DOUBLE_EQ(s, 28.0);
}

TEST(VectorTest, EpsilonEquals) {
  Vec a(std::vector<double>{1.0,2.0});
  Vec b(std::vector<double>{1.001,1.999});
  EXPECT_TRUE(a.epsilonEquals(b, 0.01));
  EXPECT_FALSE(a.epsilonEquals(b, 0.0001));
}
