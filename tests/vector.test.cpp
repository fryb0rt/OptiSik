#include "test_utils.h"
#include "vector.test.h"
#include "../data/vector.h"
#include <iostream>
#include <cmath>

using Vec = Vector<double>;

void testBasics() {
  Vec v1(std::vector<double>{1, 2, 3});
  Vec v2(std::vector<double>{4, 5, 6});
  check(close(v1.dot(v2), 32.0), "dot product");
  Vec v3 = v1 * 2.0;
  check(close(v3[0], 2.0) && close(v3[2], 6.0), "scalar multiply");
  v3 *= 0.5;
  check(v3 == v1, "in-place scalar multiply");
  Vec v4 = v2 / 2.0;
  check(close(v4[1], 2.5), "scalar divide");
}

void testArithmetic() {
  Vec a(std::vector<double>{5, 6});
  Vec b(std::vector<double>{2, 8});
  Vec c = a + b;
  check(c == Vec(std::vector<double>{7,14}), "vector add");
  Vec d = a - b;
  check(d == Vec(std::vector<double>{3,-2}), "vector sub");
  a -= b;
  check(a == Vec(std::vector<double>{3,-2}), "in-place sub");
  Vec e = 2.0 * b;
  check(e == Vec(std::vector<double>{4,16}), "scalar-left mul");
}

void testMinMax() {
  Vec x(std::vector<double>{1, 9, 3});
  Vec y(std::vector<double>{4, 2, 8});
  Vec mn = x.min(y);
  Vec mx = x.max(y);
  check(mn == Vec(std::vector<double>{1,2,3}), "element-wise min");
  check(mx == Vec(std::vector<double>{4,9,8}), "element-wise max");
  check(close(mn.minElement(), 1.0), "minElement");
  check(close(mx.maxElement(), 9.0), "maxElement");
  check(mn.minArg() == 0, "minArg");
  check(mx.maxArg() == 1, "maxArg");
}

void testNormalizeAverage() {
  Vec v(std::vector<double>{3,4});
  Vec n = v.normalized();
  check(close(n.magnitude(), 1.0, 1e-8), "normalized magnitude");
  Vec z(std::vector<double>{1,2,3,4});
  check(close(z.average(), 2.5), "average");
}

void testIteratorsFill() {
  Vec v(4);
  v.fill(7.0);
  double s = 0;
  for (auto &x : v) s += x;
  check(close(s, 28.0), "fill + iterator");
}

void testEpsilonEquals() {
  Vec a(std::vector<double>{1.0, 2.0});
  Vec b(std::vector<double>{1.001, 1.999});
  check(a.epsilonEquals(b, 0.01), "epsilonEquals true");
  check(!a.epsilonEquals(b, 0.0001), "epsilonEquals false");
}

void vectorTests() {
  std::cout << "Vector Tests" << std::endl;
  testBasics();
  testArithmetic();
  testMinMax();
  testNormalizeAverage();
  testIteratorsFill();
  testEpsilonEquals();
}
