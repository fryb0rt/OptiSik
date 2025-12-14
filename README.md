# OptiSik
Simple header-only library for optimization and equations solving.
Contents:

1) Solver for linear programming problems (optimizers/simplexSolver.h)
Solver accepts all types of constraints ( <= , ==, >= ) and floating-point or integer variables that can
be bounded or unbounded.

2) Implementation of LU materix decomposition (data/lumatrix.h)
Decomposition of matrix in lower and upper triangle. 
Allows solving of linear equations, matrix inversion and determinant computation.

3) Simple multi-dimensional gradient descent (optimizers/gradientDescent.h)
Accepts function and its gradient as functors.

3) GTests for all components - only these require compilation.

The plan is to add more features in the near future.
