#include "simplexSolver.test.h"
#include "../optimizers/simplexSolver.h"
#include "../utils/stringUtils.h"

using SS = SimplexSolver<double>;

void testCommon(const SS &solver, const SS::Operation operation,
                const double expectedCost) {
  const SS::Result result = solver.execute(operation);
  if (!solver.checkResult(result)) {
    __debugbreak();
  }
  if (abs(result.cost - expectedCost) > 1e-6) {
    std::cout << "Unexpected cost: " << result.cost << " (expected: " << expectedCost << ")" << std::endl;
  }
}

void testSolver1() {
  SS solver(3);
  solver.addConstraint(SS::Constraint{
      .coefs{3, 2, 1}, .type = SS::ConstraintType::EQUAL, .value = 10});
  solver.addConstraint(SS::Constraint{
      .coefs{2, 5, 3}, .type = SS::ConstraintType::EQUAL, .value = 15});
  solver.setVariable(0, -2);
  solver.setVariable(1, -3);
  solver.setVariable(2, -4);
  testCommon(solver, SS::Operation::MINIMIZE, -18.571428571428569);
}

void testSolver2() {
  SS solver(2);
  solver.addConstraint(SS::Constraint{
      .coefs{5, 4}, .type = SS::ConstraintType::LEQUAL, .value = 32});
  solver.addConstraint(SS::Constraint{
      .coefs{1, 2}, .type = SS::ConstraintType::LEQUAL, .value = 10});
  solver.setVariable(0, -2);
  solver.setVariable(1, -3);
  testCommon(solver, SS::Operation::MINIMIZE, -17);
}

void testSolver3() {
  SS solver(3);
  solver.addConstraint(SS::Constraint{
      .coefs{2, 1, 1}, .type = SS::ConstraintType::LEQUAL, .value = 18});
  solver.addConstraint(SS::Constraint{
      .coefs{1, 2, 2}, .type = SS::ConstraintType::LEQUAL, .value = 30});
  solver.addConstraint(SS::Constraint{
      .coefs{2, 2, 2}, .type = SS::ConstraintType::LEQUAL, .value = 24});
  solver.setVariable(0, 6);
  solver.setVariable(1, 5);
  solver.setVariable(2, 4);
  testCommon(solver, SS::Operation::MAXIMIZE, 66);
}

void testSolver4() {
  SS solver(3);
  solver.addConstraint(SS::Constraint{
      .coefs{3, 2, 1}, .type = SS::ConstraintType::LEQUAL, .value = 10});
  solver.addConstraint(SS::Constraint{
      .coefs{2, 5, 3}, .type = SS::ConstraintType::LEQUAL, .value = 15});
  solver.setVariable(0, -2);
  solver.setVariable(1, -3);
  solver.setVariable(2, -4);
  testCommon(solver, SS::Operation::MINIMIZE, -20);
}

void testSolver5() {
  SS solver(3);
  solver.addConstraint(SS::Constraint{
      .coefs{2, 5, -1}, .type = SS::ConstraintType::LEQUAL, .value = 18});
  solver.addConstraint(SS::Constraint{
      .coefs{1, -1, -2}, .type = SS::ConstraintType::LEQUAL, .value = -14});
  solver.addConstraint(SS::Constraint{
      .coefs{3, 2, 2}, .type = SS::ConstraintType::EQUAL, .value = 26});
  solver.setVariable(0, -6);
  solver.setVariable(1, 7);
  solver.setVariable(2, 4);
  testCommon(solver, SS::Operation::MINIMIZE, 16);
}

void testSolver6() {
  SS solver(2);
  solver.addConstraint(SS::Constraint{
      .coefs{1, 3}, .type = SS::ConstraintType::LEQUAL, .value = 200});
  solver.addConstraint(SS::Constraint{
      .coefs{1, 1}, .type = SS::ConstraintType::EQUAL, .value = 80});
  solver.addConstraint(SS::Constraint{
      .coefs{1, 0}, .type = SS::ConstraintType::GEQUAL, .value = 10});
  solver.addConstraint(SS::Constraint{
      .coefs{0, 1}, .type = SS::ConstraintType::GEQUAL, .value = 30});
  solver.setVariable(0, 1);
  solver.setVariable(1, 2);
  testCommon(solver, SS::Operation::MAXIMIZE, 140);
}

void testSolver7() {
  SS solver(1);
  solver.addConstraint(SS::Constraint{
      .coefs{1}, .type = SS::ConstraintType::LEQUAL, .value = 200});
  solver.setVariable(0, 2, SS::VariableBounds::UNBOUNDED);
  testCommon(solver, SS::Operation::MAXIMIZE, 400);
}

void testSolver8() {
  SS solver(2);
  solver.addConstraint(SS::Constraint{
      .coefs{1, 1}, .type = SS::ConstraintType::GEQUAL, .value = -200});
  solver.setVariable(0, 1, SS::VariableBounds::NON_POSITIVE);
  solver.setVariable(1, 1, SS::VariableBounds::NON_POSITIVE);
  testCommon(solver, SS::Operation::MINIMIZE, -200);
}

void testSolver9() {
  SS solver(2);
  solver.addConstraint(SS::Constraint{
      .coefs{1, 1}, .type = SS::ConstraintType::LEQUAL, .value = 5});
  solver.addConstraint(SS::Constraint{
      .coefs{4, 7}, .type = SS::ConstraintType::LEQUAL, .value = 28});
  solver.setVariable(0, 5, SS::VariableBounds::NON_NEGATIVE,
                     SS::VariableType::INTEGER);
  solver.setVariable(1, 6, SS::VariableBounds::NON_NEGATIVE,
                     SS::VariableType::INTEGER);
  testCommon(solver, SS::Operation::MAXIMIZE, 27);
}

void testSolver10() {
  std::string line = "[.#.###.#] (0,2,6,7) (2,3) (1,2,3,5,7) (0,1,3,4,5,6,7) "
                     "(0,1,2,3,5,7) (0,2,4,5,7) (1,2,6,7) (0,2,3,5) (3,7) "
                     "(2,3,4) {35,44,107,74,41,44,31,81}";
  auto s = split(line, ' ');
  std::vector<std::vector<int>> buttons;
  for (int i = 1; i < s.size() - 1; ++i) {
    buttons.push_back(parseList<int>(s[i], ','));
  }
  std::vector<int> values = parseList<int>(s.back(), ',');
  SS solver(int(buttons.size()));
  for (int i = 0; i < values.size(); ++i) {
    SS::Constraint c;
    c.coefs.resize(buttons.size(), 0);
    c.type = SS::ConstraintType::EQUAL;
    c.value = double(values[i]);
    for (int j = 0; j < buttons.size(); ++j) {
      bool hasOne = false;
      for (int v : buttons[j]) {
        hasOne |= v == i;
      }
      c.coefs[j] = double(int(hasOne));
    }
    solver.addConstraint(c);
  }
  for (int i = 0; i < buttons.size(); ++i) {
    solver.setVariable(i, 1.0f, SS::VariableBounds::NON_NEGATIVE,
                       SS::VariableType::INTEGER);
  }
  testCommon(solver, SS::Operation::MINIMIZE, 114);
}

void simplexSolverTests() {
  std::cout << "Simplex Solver Tests" << std::endl;
  testSolver1();
  testSolver2();
  testSolver3();
  testSolver4();
  testSolver5();
  testSolver6();
  testSolver7();
  testSolver8();
  testSolver9();
  testSolver10();
}
