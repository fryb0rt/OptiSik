#include "tests/simplexSolver.test.h"
#include "tests/vector.test.h"
#include "tests/matrix.test.h"

int main() {
  simplexSolverTests();
  vectorTests();
  matrixTests();
  return 0;
}