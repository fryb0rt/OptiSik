#include "test_utils.h"
#include "matrix.test.h"
#include "../data/matrix.h"
#include "../data/vector.h"
#include <iostream>
#include <cmath>

using Mat = Matrix<double>;
using Vec = Vector<double>;

void testBasicOps() {
  Mat m1(2, 3);
  m1(0, 0) = 1; m1(0, 1) = 2; m1(0, 2) = 3;
  m1(1, 0) = 4; m1(1, 1) = 5; m1(1, 2) = 6;
  check(m1.rows() == 2 && m1.cols() == 3, "matrix dimensions");
  check(close(m1(0, 1), 2.0), "element access");
  
  Mat m2 = m1 * 2.0;
  check(close(m2(1, 2), 12.0), "scalar multiply");
  
  Mat m3 = m1 / 2.0;
  check(close(m3(0, 0), 0.5), "scalar divide");
}

void testMatrixAddSub() {
  Mat a(2, 2);
  a(0, 0) = 1; a(0, 1) = 2;
  a(1, 0) = 3; a(1, 1) = 4;
  
  Mat b(2, 2);
  b(0, 0) = 5; b(0, 1) = 6;
  b(1, 0) = 7; b(1, 1) = 8;
  
  Mat c = a + b;
  check(close(c(0, 0), 6.0) && close(c(1, 1), 12.0), "matrix add");
  
  Mat d = b - a;
  check(close(d(0, 0), 4.0) && close(d(1, 0), 4.0), "matrix sub");
}

void testMatrixMul() {
  Mat a(2, 3);
  a(0, 0) = 1; a(0, 1) = 2; a(0, 2) = 3;
  a(1, 0) = 4; a(1, 1) = 5; a(1, 2) = 6;
  
  Mat b(3, 2);
  b(0, 0) = 7; b(0, 1) = 8;
  b(1, 0) = 9; b(1, 1) = 10;
  b(2, 0) = 11; b(2, 1) = 12;
  
  Mat c = a * b;
  check(c.rows() == 2 && c.cols() == 2, "matrix mul dims");
  check(close(c(0, 0), 58.0), "matrix mul [0,0]");
  check(close(c(0, 1), 64.0), "matrix mul [0,1]");
  check(close(c(1, 0), 139.0), "matrix mul [1,0]");
  check(close(c(1, 1), 154.0), "matrix mul [1,1]");
}

void testMatrixVecMul() {
  Mat m(2, 3);
  m(0, 0) = 1; m(0, 1) = 2; m(0, 2) = 3;
  m(1, 0) = 4; m(1, 1) = 5; m(1, 2) = 6;
  
  Vec v(std::vector<double>{1, 2, 3});
  Vec result = m * v;
  check(result.dimension() == 2, "matrix-vector mul dimension");
  check(close(result[0], 14.0), "matrix-vector mul [0]");
  check(close(result[1], 32.0), "matrix-vector mul [1]");
}

void testTranspose() {
  Mat m(2, 3);
  m(0, 0) = 1; m(0, 1) = 2; m(0, 2) = 3;
  m(1, 0) = 4; m(1, 1) = 5; m(1, 2) = 6;
  
  Mat t = m.transpose();
  check(t.rows() == 3 && t.cols() == 2, "transpose dims");
  check(close(t(0, 0), 1.0) && close(t(2, 1), 6.0), "transpose values");
}

void testIdentityTrace() {
  Mat id = Mat::identity(3);
  check(close(id.trace(), 3.0), "identity trace");
  check(close(id(0, 0), 1.0) && close(id(1, 1), 1.0), "identity values");
  
  Mat m(2, 2);
  m(0, 0) = 5; m(0, 1) = 2;
  m(1, 0) = 3; m(1, 1) = 7;
  check(close(m.trace(), 12.0), "matrix trace");
}

void testFill() {
  Mat m(2, 2);
  m.fill(3.14);
  check(close(m(0, 0), 3.14) && close(m(1, 1), 3.14), "matrix fill");
}

void testEquality() {
  Mat a(2, 2);
  a(0, 0) = 1; a(0, 1) = 2;
  a(1, 0) = 3; a(1, 1) = 4;
  
  Mat b = a;
  check(a == b, "matrix equality");
  
  b(0, 0) = 99;
  check(a != b, "matrix inequality");
}

void matrixTests() {
  std::cout << "Matrix Tests" << std::endl;
  testBasicOps();
  testMatrixAddSub();
  testMatrixMul();
  testMatrixVecMul();
  testTranspose();
  testIdentityTrace();
  testFill();
  testEquality();
}
