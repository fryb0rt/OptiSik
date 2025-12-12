#pragma once
#include <iostream>
#include <cmath>

static bool close(double a, double b, double eps = 1e-9) {
  return std::abs(a - b) <= eps;
}

static void check(bool cond, const char* msg) {
  if (!cond) {
    std::cout << "ERROR: " << msg << std::endl;
    __debugbreak();
  }
}
