cmake_minimum_required(VERSION 3.10.0)
project(OptiSik VERSION 0.1.0 LANGUAGES C CXX)

add_executable(OptiSik test.cpp tests/simplexSolver.test.cpp tests/vector.test.cpp tests/matrix.test.cpp)

