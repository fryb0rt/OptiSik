#include <gtest/gtest.h>
#include "data/matrix.h"
#include "data/vector.h"

using namespace OptiSik;
using Mat = Matrix<double>;
using Vec = Vector<double>;

TEST(MatrixTest, BasicOps) {
  Mat m1(2,3);
  m1(0,0)=1; m1(0,1)=2; m1(0,2)=3;
  m1(1,0)=4; m1(1,1)=5; m1(1,2)=6;
  EXPECT_EQ(m1.rows(), 2u);
  EXPECT_EQ(m1.cols(), 3u);
  EXPECT_DOUBLE_EQ(m1(0,1), 2.0);
  Mat m2 = m1 * 2.0;
  EXPECT_DOUBLE_EQ(m2(1,2), 12.0);
  Mat m3 = m1 / 2.0;
  EXPECT_DOUBLE_EQ(m3(0,0), 0.5);
}

TEST(MatrixTest, AddSub) {
  Mat a(2,2);
  a(0,0)=1; a(0,1)=2; a(1,0)=3; a(1,1)=4;
  Mat b(2,2);
  b(0,0)=5; b(0,1)=6; b(1,0)=7; b(1,1)=8;
  Mat c = a + b;
  EXPECT_DOUBLE_EQ(c(0,0), 6.0);
  Mat d = b - a;
  EXPECT_DOUBLE_EQ(d(0,0), 4.0);
}

TEST(MatrixTest, Mul) {
  Mat a(2,3);
  a(0,0)=1; a(0,1)=2; a(0,2)=3;
  a(1,0)=4; a(1,1)=5; a(1,2)=6;
  Mat b(3,2);
  b(0,0)=7; b(0,1)=8; b(1,0)=9; b(1,1)=10; b(2,0)=11; b(2,1)=12;
  Mat c = a * b;
  EXPECT_EQ(c.rows(), 2u);
  EXPECT_EQ(c.cols(), 2u);
  EXPECT_DOUBLE_EQ(c(0,0), 58.0);
  EXPECT_DOUBLE_EQ(c(1,1), 154.0);
}

TEST(MatrixTest, MatrixVector) {
  Mat m(2,3);
  m(0,0)=1; m(0,1)=2; m(0,2)=3;
  m(1,0)=4; m(1,1)=5; m(1,2)=6;
  Vec v(std::vector<double>{1,2,3});
  Vec r = m * v;
  EXPECT_EQ(r.dimension(), 2u);
  EXPECT_DOUBLE_EQ(r[0], 14.0);
}

TEST(MatrixTest, TransposeTraceIdentity) {
  Mat m(2,3);
  m(0,0)=1; m(0,1)=2; m(0,2)=3;
  m(1,0)=4; m(1,1)=5; m(1,2)=6;
  Mat t = m.transpose();
  EXPECT_EQ(t.rows(), 3u);
  EXPECT_DOUBLE_EQ(t(2,1), 6.0);
  Mat id = Mat::identity(3);
  EXPECT_DOUBLE_EQ(id.trace(), 3.0);
}

TEST(MatrixTest, ConstructFromVectorOfVectors) {
  std::vector<std::vector<double>> data{{1, 2}, {3, 4}};
  Mat m(data);
  EXPECT_EQ(m.rows(), 2u);
  EXPECT_EQ(m.cols(), 2u);
  EXPECT_DOUBLE_EQ(m(0,0), 1.0);
  EXPECT_DOUBLE_EQ(m(1,1), 4.0);
}

TEST(MatrixTest, ConstructFromVectorOfVectorsOfVectorObjects) {
  std::vector<Vec> rows;
  rows.emplace_back(std::vector<double>{1,2});
  rows.emplace_back(std::vector<double>{3,4});
  Mat m(rows);
  Mat expected(std::vector<std::vector<double>>{{1,2},{3,4}});
  EXPECT_TRUE(m == expected);
}

TEST(MatrixTest, EqualityDifferentConstructors) {
  Mat a(2,2);
  a(0,0)=1; a(0,1)=2; a(1,0)=3; a(1,1)=4;
  std::vector<Vec> rows;
  rows.emplace_back(std::vector<double>{1,2});
  rows.emplace_back(std::vector<double>{3,4});
  Mat b(rows);
  EXPECT_EQ(a, b);
}

TEST(MatrixTest, EpsilonEquals) {
  Mat a(std::vector<std::vector<double>>{{1.0,2.0},{3.0,4.0}});
  Mat b(std::vector<std::vector<double>>{{1.001,1.999},{3.0005,4.0}});
  EXPECT_TRUE(a.epsilonEquals(b, 0.01));
  EXPECT_FALSE(a.epsilonEquals(b, 0.0001));
}
